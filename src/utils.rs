pub fn focal_len_find(dist: f32, width: f32, rf_width: u32) -> f32 {
    (rf_width as f32 * dist) / width
}

pub fn find_dist(focal_len: f32, known_width: f32, frame_width: u32) -> f32 {
    (known_width * focal_len) / frame_width as f32
}
