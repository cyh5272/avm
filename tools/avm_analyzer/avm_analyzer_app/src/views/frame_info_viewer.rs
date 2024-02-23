use avm_stats::Frame;
use egui::{RichText, Ui};
use egui_extras::{Column, TableBuilder};

use crate::{app_state::AppState, stream::CurrentFrame};

use crate::views::render_view::RenderView;

// TODO(comc): PSNR, SSIM stats if available.
pub struct FrameInfoViewer;
impl FrameInfoViewer {
    fn frame_info(frame: &Frame) -> Option<Vec<(String, String)>> {
        let mut info = Vec::new();

        let params = frame.frame_params.as_ref()?;
        info.push(("Stream".into(), frame.stream_params.as_ref()?.stream_name.to_string()));
        info.push(("Width".into(), format!("{}", params.width)));
        info.push(("Height".into(), format!("{}", params.height)));
        info.push(("Frame Type".into(), frame.frame_type_name()));
        info.push(("TIP mode".into(), frame.tip_mode_name()));
        info.push(("Decode Index".into(), params.decode_index.to_string()));
        info.push(("Display Index".into(), params.display_index.to_string()));
        if let Some(superblock_size) = params.superblock_size.as_ref() {
            info.push((
                "Superblock size".into(),
                format!("{}x{}", superblock_size.width, superblock_size.height),
            ));
        }
        // TODO(comc): Check why this is false for inter frames.
        // info.push(("Show frame".into(), params.show_frame.to_string()));
        info.push(("Base QIndex".into(), params.base_qindex.to_string()));
        info.push(("Bit depth".into(), params.bit_depth.to_string()));
        Some(info)
    }
}
impl RenderView for FrameInfoViewer {
    fn title(&self) -> String {
        "Frame Info".into()
    }

    fn render(&self, ui: &mut Ui, state: &mut AppState) -> anyhow::Result<()> {
        let Some(frame) = state.stream.current_frame() else {
            return Ok(());
        };

        let Some(frame_info) = Self::frame_info(frame) else {
            return Ok(());
        };
        TableBuilder::new(ui)
            .column(Column::initial(300.0).resizable(true))
            .column(Column::remainder())
            .striped(true)
            .header(20.0, |mut header| {
                header.col(|ui| {
                    ui.label(RichText::new("Property").strong());
                });
                header.col(|ui| {
                    ui.label(RichText::new("Value").strong());
                });
            })
            .body(|mut body| {
                for (info_name, info_value) in frame_info {
                    body.row(20.0, |mut row| {
                        row.col(|col| {
                            col.label(info_name);
                        });
                        row.col(|col| {
                            col.label(info_value);
                        });
                    });
                }
            });
        Ok(())
    }
}
