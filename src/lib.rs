use std::error::Error;

/// convert an index to a column-major subscript given an array shape
#[inline(always)]
pub fn idx_to_sub_col_major(
    index: usize,
    shape: &[usize],
    subscript: &mut [usize],
) -> Result<(), Box<dyn Error>> {
    let remaining = index;
    let mut stride = 1;
    for (sub, size) in subscript.iter_mut().zip(shape.iter()) {
        *sub = remaining / stride % size;
        if *sub >= *size {
            Err(format!(
                "subscript {} is greater than or equal to dim {}",
                *sub, *size
            ))?
        }
        stride *= size;
    }
    Ok(())
}

pub fn strides_col_major(dims: &[usize]) -> Vec<usize> {
    let mut strides = vec![0; dims.len()];
    calc_strides_col_major(&dims, &mut strides);
    strides
}

pub fn calc_strides_col_major(dims: &[usize], strides: &mut [usize]) {
    let mut stride = 1;
    strides.iter_mut().zip(dims).for_each(|(s, &d)| {
        *s = stride;
        stride *= d
    });
}

pub fn sub_to_idx_col_major(subscript: &[usize], shape: &[usize]) -> Result<usize, Box<dyn Error>> {
    if subscript.len() != shape.len() {
        Err("Sizes of subscripts and shape do not match")?
    }

    let mut index = 0;
    let mut stride = 1;

    for (sub, size) in subscript.iter().zip(shape.iter()) {
        if *sub >= *size {
            Err(format!(
                "subscript {} is greater than or equal to dim {}",
                *sub, *size
            ))?
        }
        index += sub * stride;
        stride *= size;
    }

    Ok(index)
}

pub fn idx_to_coord_col_major(
    index: usize,
    shape: &[usize],
    subscript: &mut [usize],
    coord: &mut [i32],
) -> Result<(), Box<dyn Error>> {
    idx_to_sub_col_major(index, shape, subscript)?;
    for ((sub, c), dim) in subscript.iter().zip(coord.iter_mut()).zip(shape.iter()) {
        *c = ((*sub) as i32) - (dim / 2) as i32;
    }
    Ok(())
}

pub fn coord_to_idx_col_major(
    coord: &[i32],
    shape: &[usize],
    scratch_space: &mut [usize],
) -> Result<usize, Box<dyn Error>> {
    for ((c, sub), dim) in coord.iter().zip(scratch_space.iter_mut()).zip(shape.iter()) {
        *sub = (c + (dim / 2) as i32).try_into()?;
    }
    sub_to_idx_col_major(&scratch_space, shape)
}

// right permute
pub fn idx_map(idx: usize, shape: &[usize]) -> usize {
    let mut shape = shape.to_owned();
    let mut buff = vec![0; shape.len()];
    idx_to_sub_col_major(idx, &shape, &mut buff).unwrap();
    shape.rotate_right(1);
    buff.rotate_right(1);
    sub_to_idx_col_major(&buff, &shape).unwrap()
}

mod tests {
    use super::*;

    #[test]
    fn calc_strides() {
        let dims = vec![2, 3, 4, 5];
        let mut expected_strides = vec![1, 2, 6, 24];
        let returned_strides = strides_col_major(&dims);
        assert_eq!(expected_strides, returned_strides);

        calc_strides_col_major(&dims, &mut expected_strides);
        assert_eq!(expected_strides, returned_strides);
    }
}
