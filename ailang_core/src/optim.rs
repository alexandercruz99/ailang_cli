use crate::param::Param;

pub fn sgd_step(params: &mut [Param], lr: f32) {
    for param in params {
        param.sgd_step(lr);
    }
}
