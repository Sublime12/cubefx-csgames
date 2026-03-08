mod cube;

use crate::cube::{irfft, phase_shift, phase_shift_kernel_rfft_phaseshifted, rfft};
use cubecl::{CubeCount, CubeDim, Runtime, ir::StorageType, prelude::InputScalar, std::tensor::TensorHandle};

pub struct SignalSpec {
    pub signal_duration: f32,
    pub channels: usize,
    pub sample_rate: usize,
    pub window_length: usize,
    pub hop_length: usize,
}

impl SignalSpec {
    pub fn signal_shape(&self) -> [usize; 3] {
        let total_samples = (self.signal_duration * self.sample_rate as f32).ceil() as usize;
        let num_windows = total_samples.div_ceil(self.hop_length);
        [num_windows, self.channels, self.window_length]
    }

    pub fn spectrum_shape(&self) -> [usize; 3] {
        let total_samples = (self.signal_duration * self.sample_rate as f32).ceil() as usize;
        let num_windows = total_samples.div_ceil(self.hop_length);
        let num_frequency_bins = self.window_length / 2 + 1;
        [num_windows, self.channels, num_frequency_bins]
    }
}

pub fn phase_shift_effect<R: Runtime>(
    signal: TensorHandle<R>,
    alpha: f32,
    dtype: StorageType,
) -> TensorHandle<R> {
    // let (spectrum_re, spectrum_im) = rfft(signal, dtype);
    let dim = signal.shape.len() - 1;
    assert!(
        signal.shape[dim].is_power_of_two(),
        "RFFT requires power-of-2 length"
    );
    let client = <R as Runtime>::client(&Default::default());

    let mut spectrum_shape = signal.shape.clone();
    spectrum_shape[dim] = signal.shape[dim] / 2 + 1;

    let spectrum_re = TensorHandle::new_contiguous(
        spectrum_shape.clone(),
        client.empty(spectrum_shape.iter().product::<usize>() * dtype.size()),
        dtype,
    );

    let spectrum_im = TensorHandle::new_contiguous(
        spectrum_shape.clone(),
        client.empty(spectrum_shape.iter().product::<usize>() * dtype.size()),
        dtype,
    );

    let cube_count = CubeCount::new_2d(signal.shape[0] as u32, signal.shape[1] as u32);
    let cube_dim = CubeDim::new_single();
    let vectorization = 1;

    // rfft_kernel::launch::<R>(
    //     &client,
    //     cube_count,
    //     cube_dim,
    //     signal.as_tensor_arg(vectorization),
    //     spectrum_re.as_tensor_arg(vectorization),
    //     spectrum_im.as_tensor_arg(vectorization),
    //     *signal.shape.last().unwrap(),
    //     dtype,
    // );

    // let (shifted_re, shifted_im) = phase_shift(spectrum_re, spectrum_im, alpha);

    let client = <R as Runtime>::client(&Default::default());
    let shape = spectrum_re.shape.clone();
    let num_elements = shape.iter().product::<usize>();
    let dtype = spectrum_re.dtype;

    let output_re: TensorHandle<R> = TensorHandle::new_contiguous(
        shape.clone(),
        client.empty(num_elements * dtype.size()),
        dtype,
    );

    let output_im: TensorHandle<R> =
        TensorHandle::new_contiguous(shape, client.empty(num_elements * dtype.size()), dtype);

    phase_shift_kernel_rfft_phaseshifted::launch(
        &client,
        cube_count,
        cube_dim,
        signal.as_ref().as_tensor_arg(vectorization),
        spectrum_re.as_ref().as_tensor_arg(vectorization),
        spectrum_im.as_ref().as_tensor_arg(vectorization),
        output_re.as_ref().as_tensor_arg(vectorization),
        output_im.as_ref().as_tensor_arg(vectorization),
        InputScalar::new(alpha, dtype),
        dtype,
        *signal.shape.last().unwrap(),
    ).unwrap();
    irfft(output_re, output_im, dtype)
}
