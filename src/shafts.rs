//! The sun's light carved out of the air wherever the terrain or a cloud stands
//! between that air and the sun: the god rays past a peak, and the shadows a
//! low sun lays across the haze.
//!
//! What is computed and why it is computed this way is said at the top of
//! `src/shafts.wgsl`. This owns the two half-resolution passes -- the march,
//! and the blur that takes its interleave back out -- and the buffers between
//! them and the shading.

use crate::deferred::GBuffer;

/// Threads per side of a workgroup of either pass. Must match `SHAFT_GROUP` in
/// `src/shafts.wgsl`.
const SHAFT_GROUP: u32 = 8;

/// The format of everything this writes: three channels of light lost, which
/// spans the sky's range and wants more than eight bits, and one more number.
const SHAFT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// The format of the share of what is lost in front of the cloud: one channel,
/// and thirty-two bits because the sixteen-bit float formats cannot be written
/// as storage without a feature.
const SHARE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;

/// One set of the march's outputs: the light lost out to a block's farthest
/// pixel and out to its nearest, and the share of the first in front of the
/// cloud. See `out_far` in `src/shafts.wgsl`.
struct Pair {
    #[allow(dead_code, reason = "read by the tests; the views keep them alive")]
    far: wgpu::Texture,
    #[allow(dead_code, reason = "read by the tests; the views keep them alive")]
    near: wgpu::Texture,
    far_view: wgpu::TextureView,
    near_view: wgpu::TextureView,
    share_view: wgpu::TextureView,
}

impl Pair {
    fn new(device: &wgpu::Device, size: glam::UVec2, label: &str) -> Self {
        let texture = |format| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: size.x,
                    height: size.y,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            })
        };
        let (far, near) = (texture(SHAFT_FORMAT), texture(SHAFT_FORMAT));
        let far_view = far.create_view(&Default::default());
        let near_view = near.create_view(&Default::default());
        let share_view = texture(SHARE_FORMAT).create_view(&Default::default());
        Self {
            far,
            near,
            far_view,
            near_view,
            share_view,
        }
    }
}

/// What the shading reads: the blurred pair.
#[derive(Clone, Copy)]
pub struct Lost<'a> {
    pub far: &'a wgpu::TextureView,
    pub near: &'a wgpu::TextureView,
    pub share: &'a wgpu::TextureView,
}

pub struct Shafts {
    march_layout: wgpu::BindGroupLayout,
    blur_layout: wgpu::BindGroupLayout,
    march_pipeline: wgpu::ComputePipeline,
    blur_pipeline: wgpu::ComputePipeline,
    size: glam::UVec2,
    /// What the march writes and the blur reads.
    raw: Pair,
    /// What the blur writes and the shading reads.
    blurred: Pair,
    /// One per parity of the cloud buffer, whose distances say where the
    /// cloud on each ray is; see [`crate::cloud::March::parity`].
    march_groups: [wgpu::BindGroup; 2],
    blur_group: wgpu::BindGroup,
}

impl Shafts {
    /// Builds both passes against the layouts their owners hold, at half the
    /// size of `gbuffer`.
    pub fn new(
        device: &wgpu::Device,
        camera_layout: &wgpu::BindGroupLayout,
        sky_layout: &wgpu::BindGroupLayout,
        sun_tables_layout: &wgpu::BindGroupLayout,
        gbuffer: &GBuffer,
        clouds: crate::deferred::Clouds<'_>,
        shadow: crate::shadow::Field<'_>,
    ) -> Self {
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty,
            count: None,
        };
        let texture = |sample_type, view_dimension| wgpu::BindingType::Texture {
            sample_type,
            view_dimension,
            multisampled: false,
        };
        let unfiltered = wgpu::TextureSampleType::Float { filterable: false };
        let uniform = wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let written = wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format: SHAFT_FORMAT,
            view_dimension: wgpu::TextureViewDimension::D2,
        };
        let written_share = wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format: SHARE_FORMAT,
            view_dimension: wgpu::TextureViewDimension::D2,
        };
        let march_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shafts march layout"),
            entries: &[
                entry(0, texture(unfiltered, wgpu::TextureViewDimension::D2)),
                entry(1, texture(unfiltered, wgpu::TextureViewDimension::D2)),
                entry(
                    2,
                    texture(
                        wgpu::TextureSampleType::Float { filterable: true },
                        wgpu::TextureViewDimension::D3,
                    ),
                ),
                entry(3, uniform),
                entry(
                    4,
                    texture(wgpu::TextureSampleType::Uint, wgpu::TextureViewDimension::D2Array),
                ),
                entry(5, uniform),
                entry(6, written),
                entry(7, written),
                entry(10, written_share),
            ],
        });
        // The blur's own, which leaves out the pair the march writes: the blur
        // reads that pair, and a texture may not be bound both ways at once.
        let blur_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shafts blur layout"),
            entries: &[
                entry(6, written),
                entry(7, written),
                entry(10, written_share),
                entry(8, texture(unfiltered, wgpu::TextureViewDimension::D2)),
                entry(9, texture(unfiltered, wgpu::TextureViewDimension::D2)),
                entry(11, texture(unfiltered, wgpu::TextureViewDimension::D2)),
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shafts shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shafts.wgsl").into()),
        });
        let pipeline = |label, entry_point, layouts: &[Option<&wgpu::BindGroupLayout>]| {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: layouts,
                immediate_size: 0,
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                module: &shader,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let march_pipeline = pipeline(
            "shafts march pipeline",
            "cs_shafts",
            &[
                Some(camera_layout),
                Some(sky_layout),
                Some(sun_tables_layout),
                Some(&march_layout),
            ],
        );
        let blur_pipeline = pipeline(
            "shafts blur pipeline",
            "cs_shafts_blur",
            &[None, None, None, Some(&blur_layout)],
        );

        let size = crate::cloud::half_of(gbuffer.size);
        let raw = Pair::new(device, size, "shafts");
        let blurred = Pair::new(device, size, "shafts blurred");
        let march_groups = Self::march_bind(device, &march_layout, gbuffer, clouds, shadow, &raw);
        let blur_group = Self::blur_bind(device, &blur_layout, &raw, &blurred);
        Self {
            march_layout,
            blur_layout,
            march_pipeline,
            blur_pipeline,
            size,
            raw,
            blurred,
            march_groups,
            blur_group,
        }
    }

    /// Follows the frame to a new size: the buffers are half of it, and the
    /// depth and the cloud buffer they read were remade with it.
    pub fn resize(
        &mut self,
        device: &wgpu::Device,
        gbuffer: &GBuffer,
        clouds: crate::deferred::Clouds<'_>,
        shadow: crate::shadow::Field<'_>,
    ) {
        self.size = crate::cloud::half_of(gbuffer.size);
        self.raw = Pair::new(device, self.size, "shafts");
        self.blurred = Pair::new(device, self.size, "shafts blurred");
        self.march_groups =
            Self::march_bind(device, &self.march_layout, gbuffer, clouds, shadow, &self.raw);
        self.blur_group = Self::blur_bind(device, &self.blur_layout, &self.raw, &self.blurred);
    }

    fn march_bind(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        gbuffer: &GBuffer,
        clouds: crate::deferred::Clouds<'_>,
        shadow: crate::shadow::Field<'_>,
        raw: &Pair,
    ) -> [wgpu::BindGroup; 2] {
        let view = |binding, view| wgpu::BindGroupEntry {
            binding,
            resource: wgpu::BindingResource::TextureView(view),
        };
        std::array::from_fn(|parity| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("shafts march group"),
                layout,
                entries: &[
                    view(0, &gbuffer.depth),
                    view(1, clouds.along[parity]),
                    view(2, clouds.volume),
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: clouds.light.as_entire_binding(),
                    },
                    view(4, shadow.view),
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: shadow.uniform.as_entire_binding(),
                    },
                    view(6, &raw.far_view),
                    view(7, &raw.near_view),
                    view(10, &raw.share_view),
                ],
            })
        })
    }

    fn blur_bind(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        raw: &Pair,
        blurred: &Pair,
    ) -> wgpu::BindGroup {
        let view = |binding, view| wgpu::BindGroupEntry {
            binding,
            resource: wgpu::BindingResource::TextureView(view),
        };
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shafts blur group"),
            layout,
            entries: &[
                view(6, &blurred.far_view),
                view(7, &blurred.near_view),
                view(10, &blurred.share_view),
                view(8, &raw.far_view),
                view(9, &raw.near_view),
                view(11, &raw.share_view),
            ],
        })
    }

    /// Records the march, after the cloud resolve whose distances it reads.
    ///
    /// Group 0, the camera, is set by the caller, as for every pass. `parity`
    /// is which of the cloud buffers this frame resolved into.
    pub fn draw(&self, pass: &mut wgpu::ComputePass<'_>, sky: &crate::sky::Sky, parity: usize) {
        pass.set_pipeline(&self.march_pipeline);
        pass.set_bind_group(1, sky.bind_group(), &[]);
        pass.set_bind_group(2, sky.sun_tables_bind_group(), &[]);
        pass.set_bind_group(3, &self.march_groups[parity], &[]);
        pass.dispatch_workgroups(
            self.size.x.div_ceil(SHAFT_GROUP),
            self.size.y.div_ceil(SHAFT_GROUP),
            1,
        );
    }

    /// Records the blur, in the pass after the march.
    pub fn draw_blur(&self, pass: &mut wgpu::ComputePass<'_>) {
        pass.set_pipeline(&self.blur_pipeline);
        pass.set_bind_group(3, &self.blur_group, &[]);
        pass.dispatch_workgroups(
            self.size.x.div_ceil(SHAFT_GROUP),
            self.size.y.div_ceil(SHAFT_GROUP),
            1,
        );
    }

    /// The blurred pair, for the shading.
    pub fn lost(&self) -> Lost<'_> {
        Lost {
            far: &self.blurred.far_view,
            near: &self.blurred.near_view,
            share: &self.blurred.share_view,
        }
    }

    /// The blurred pair's textures and their size, for the tests that read
    /// them back.
    #[cfg(test)]
    pub fn textures(&self) -> (&wgpu::Texture, &wgpu::Texture, glam::UVec2) {
        (&self.blurred.far, &self.blurred.near, self.size)
    }
}

#[cfg(test)]
mod tests {
    /// Whitespace-normalised source of a function, from its name to the next
    /// line that starts with a closing brace, comments dropped.
    fn body(source: &str, name: &str) -> String {
        let start = source
            .find(&format!("fn {name}("))
            .unwrap_or_else(|| panic!("no {name}"));
        let end = source[start..].find("\n}").expect("unterminated");
        source[start..start + end]
            .lines()
            .map(|line| line.split("//").next().unwrap_or(""))
            .flat_map(str::split_whitespace)
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn constant(source: &str, name: &str) -> Option<String> {
        source
            .lines()
            .find(|line| line.starts_with(&format!("const {name}:")))
            .map(str::to_string)
    }

    /// The air the shafts take light out of is the air `cs_aerial` put it into:
    /// the same medium, the same phase functions, the same sun through the same
    /// table. Anything else and a shadowed pixel would lose more or less than
    /// the haze ever gave it -- a shaft darker than no air at all, or one that
    /// never quite goes dark.
    #[test]
    fn the_shafts_take_out_what_the_aerial_volume_put_in() {
        let shafts = include_str!("shafts.wgsl");
        let sky = include_str!("sky.wgsl");
        let shading = include_str!("shading.wgsl");
        for name in ["medium", "rayleigh_phase", "mie_phase", "ground_distance"] {
            assert_eq!(body(shafts, name), body(sky, name), "{name} differs from src/sky.wgsl");
        }
        for name in [
            "RAYLEIGH_SCATTERING",
            "RAYLEIGH_SCALE_HEIGHT",
            "MIE_SCATTERING",
            "MIE_EXTINCTION",
            "MIE_SCALE_HEIGHT",
            "MIE_G",
            "OZONE_ABSORPTION",
            "OZONE_CENTRE",
            "OZONE_HALF_WIDTH",
            "GROUND_RADIUS",
            "TOP_RADIUS",
            "AERIAL_FAR",
        ] {
            assert_eq!(constant(shafts, name), constant(sky, name), "{name}");
        }
        // ... and it addresses the frame, the table and the cloud's light
        // volume the way the shading does.
        for name in [
            "ray_raw_at",
            "distance_at",
            "to_texture",
            "top_distance",
            "transmittance_uv",
            "sample_transmittance",
            "light_uvw",
            "stacked_w",
            "beyond_light",
            "reaching",
            "sun_reaching",
            "shadow_texel",
            "shadow_tap",
            "sun_clear",
        ] {
            assert_eq!(
                body(shafts, name),
                body(shading, name),
                "{name} differs from src/shading.wgsl"
            );
        }
        for name in [
            "LIGHT_ACROSS",
            "LIGHT_SLICES",
            "LIGHT_CASCADES",
            "CASCADE_EDGE",
            "CASCADE_FADE",
            "SHADOW_ACROSS",
            "SHADOW_CASCADES",
            "SHADOW_EDGE",
            "SHADOW_FADE",
            "SHADOW_SOFTEST",
        ] {
            assert_eq!(constant(shafts, name), constant(shading, name), "{name}");
        }
        assert_eq!(
            constant(shafts, "SHAFT_GROUP"),
            Some(format!("const SHAFT_GROUP: u32 = {}u;", super::SHAFT_GROUP))
        );
        // The top the march stops at is the top of the cloud's light volume.
        assert_eq!(
            constant(shafts, "SHADOW_TOP"),
            Some(format!("const SHADOW_TOP: f32 = {:.1};", crate::cloud::CEILING_TOP))
        );
    }
}
