use std::{thread, time::{Duration, Instant}};

use opencv::{
	core,
	highgui,
	imgproc,
	objdetect,
	prelude::*,
	Result,
	types,
	videoio,
};
mod utils;
use utils::*;

const SIZE: u32 = 288;
const DISTANCE: f32 = 393.7;
const REAL_SIZE: f32 = 177.8;

fn main() -> Result<()> {
	highgui::named_window("seancv", 1)?;
	#[cfg(not(ocvrs_opencv_branch_32))]
	let (xml, mut cam) = {
		(
			core::find_file("haarcascades/haarcascade_frontalface_alt.xml", true, false)?,
			videoio::VideoCapture::new(0, videoio::CAP_ANY)?, // 0 is the default camera
		)
	};
    cam.set(3, 1920.0)?;
    cam.set(4, 1080.0)?;
	let _ = videoio::VideoCapture::is_opened(&cam)?;
	let mut face = objdetect::CascadeClassifier::new(&xml)?;
    let mut frame_count: usize = 0;
    let startup = Instant::now();
    let focal = focal_len_find(DISTANCE, REAL_SIZE, SIZE);
	loop {
        frame_count += 1;
		let mut frame = Mat::default();
		cam.read(&mut frame)?;
		if frame.size()?.width == 0 || frame.size()?.height == 0 {
			thread::sleep(Duration::from_millis(50));
			continue;
		}
		let mut gray = Mat::default();
		imgproc::cvt_color(
			&frame,
			&mut gray,
			imgproc::COLOR_BGR2GRAY,
			0,
		)?;
		let mut reduced = Mat::default();
		imgproc::resize(
			&gray,
			&mut reduced,
			core::Size {
				width: 0,
				height: 0,
			},
			0.25f64,
			0.25f64,
			imgproc::INTER_LINEAR,
		)?;
		let mut faces = types::VectorOfRect::new();
		face.detect_multi_scale(
			&reduced,
			&mut faces,
			1.1,
			2,
			objdetect::CASCADE_SCALE_IMAGE,
			core::Size {
				width: 30,
				height: 30,
			},
			core::Size {
				width: 0,
				height: 0,
			},
		)?;
        //if faces.len() == 0 { continue; }
		for face in faces {
            //let distance = 
			print!(
                "\rface {:?}\t{:.2} fps \t{} mm   ", 
                face, 
                frame_count as f32 / startup.elapsed().as_secs() as f32,
                find_dist(focal, REAL_SIZE, face.width as u32 * 4),
            );
            //println!("{:?} {:?}", face.width, face.height);
			let scaled_face = core::Rect {
				x: face.x * 4,
				y: face.y * 4,
				width: face.width * 4,
				height: face.height * 4,
			};
			imgproc::rectangle(
				&mut frame,
				scaled_face,
				core::Scalar::new(255f64, 0f64, 0f64, 25f64), // color value
				1, // border width of drawn rectangle
				8,
				0,
			)?;
		}
		highgui::imshow("seancv", &frame)?;
		if highgui::wait_key(10)? > 0 {
            continue;
		}
	}
	//Ok(())
}
