extern crate num_complex;
use self::num_complex::Complex;
use vec_to_mat;
fn help_compute_moments(
    alpha:f64,
    t:f64
)->f64{
    (1.0-(-alpha*t).exp())/alpha
}
fn cross_multiply(
    rho:f64,
    sigma_1:f64,
    sigma_2:f64,
    alpha_1:f64,
    alpha_2:f64
)->f64{
    (rho*sigma_1*sigma_2)/(alpha_1*alpha_2)
}
pub fn compute_expectation(
    y0:&[f64],
    alpha:&[f64],
    beta:&[f64],
    t:f64
)->Vec<f64> 
{
    y0.iter().zip(alpha.iter().zip(beta.iter())).map(|(y_init, (alpha_item, beta_item))|{
        (*y_init-*beta_item)*((-*alpha_item*t).exp())+*beta_item
    }).collect()
}

pub fn compute_expectation_long_run_one(
    y0:&[f64],
    alpha:&[f64],
    t:f64
)->Vec<f64>{
    y0.iter().zip(alpha.iter()).map(|(y_init, alpha_item)|{
        (*y_init-1.0)*((-*alpha_item*t).exp())+1.0
    }).collect()
}

pub fn compute_integral_expectation(
    y0:&[f64],
    alpha:&[f64],
    beta:&[f64],
    t:f64
)->Vec<f64>{
    y0.iter().zip(alpha.iter().zip(beta.iter())).map(|(&y_init, (&alpha_item, &beta_item))|{
        (y_init-beta_item)*help_compute_moments(alpha_item, t)+beta_item*t
    }).collect()
}

pub fn compute_integral_expectation_long_run_one(
    y0:&[f64],
    alpha:&[f64],
    t:f64
)->Vec<f64>{
    y0.iter().zip(alpha.iter()).map(|(&y_init, &alpha_item)|{
        (y_init-1.0)*help_compute_moments(alpha_item, t)+t
    }).collect()
}

fn get_one_d_array_outer_loop<T>(
    index:usize, 
    num_rows:usize,
    array:&[T]
)->&T {
    &array[vec_to_mat::get_row_from_index(index, num_rows)]
}
fn get_one_d_array_inner_loop<T>(
    index:usize, 
    num_rows:usize,
    array:&[T]
)->&T {
    &array[vec_to_mat::get_col_from_index(index, num_rows)] //rounds down
}
pub fn compute_integral_variance(
    alpha:&[f64],
    sigma:&[f64],
    rho:&[f64],
    t:f64
)->Vec<f64>{
    let num_rows=alpha.len();
    alpha.iter().zip(sigma).enumerate().flat_map(|(i, (&alpha_elem_i, &sigma_elem_i))|{
        let a_i=help_compute_moments(alpha_elem_i, t);
        alpha.iter().zip(sigma).enumerate().map(move |(j, (&alpha_elem_j, &sigma_elem_j))|{
            let a_j=help_compute_moments(alpha_elem_j, t);
            cross_multiply(
                *vec_to_mat::get_element_from_matrix(
                    i, j, num_rows, rho
                ),
                sigma_elem_i,
                sigma_elem_j, 
                alpha_elem_i,
                alpha_elem_j
            )*(t-a_i-a_j+help_compute_moments(alpha_elem_i+alpha_elem_j, t))
        })
    }).collect()
}

pub fn get_log_vasicek_mgf(
    expectation:Vec<f64>,
    variance:Vec<f64>
)->impl Fn(&[Complex<f64>])->Complex<f64>{
    let num_rows=expectation.len();
    move |u_vec|{
        expectation.iter().zip(u_vec).map(|(exp_increment, u_increment)|{
            exp_increment*u_increment
        }).sum::<Complex<f64>>()+variance.iter().enumerate().map(|(index, var_increment)|{
            var_increment
                *get_one_d_array_inner_loop(index, num_rows, u_vec)
                *get_one_d_array_outer_loop(index, num_rows, u_vec)
        }).sum::<Complex<f64>>()*0.5
    }
}

pub fn get_vasicek_mgf(
    expectation:Vec<f64>,
    variance:Vec<f64>
)->impl Fn(&[Complex<f64>])->Complex<f64>{
    let log_vasicek_mgf=get_log_vasicek_mgf(expectation, variance);
    move |u_vec|log_vasicek_mgf(u_vec).exp()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_get_one_d_array_outer(){
        let arr=vec![1, 2, 3];
        let num_cols=3;
        assert_eq!(*get_one_d_array_outer_loop(4, num_cols, &arr), 2);
        assert_eq!(*get_one_d_array_outer_loop(0, num_cols, &arr), 1);
    }
    #[test]
    fn test_get_one_d_array_inner(){
        let arr=vec![1, 2, 3];
        let num_cols=3;
        assert_eq!(*get_one_d_array_inner_loop(4, num_cols, &arr), 2);
        assert_eq!(*get_one_d_array_inner_loop(2, num_cols, &arr), 1);
        assert_eq!(*get_one_d_array_inner_loop(0, num_cols, &arr), 1);
    }
    #[test]
    fn test_compute_expectation_long_run(){
        let y0=vec![0.5];
        let alpha=vec![0.3];
        let beta=vec![0.5];
        let t=1.0;
        assert_eq!(*compute_expectation(&y0, &alpha, &beta, t).first().unwrap(), beta[0]);
    }
    #[test]
    fn test_combine_inner_and_outer(){
        let u_array_test:Vec<u32>=vec![0, 1, 2];
        let num_cols=u_array_test.len();
        let expected_outer:Vec<u32>=vec![0, 1, 2, 0, 1, 2, 0, 1, 2];
        let expected_inner:Vec<u32>=vec![0, 0, 0, 1, 1, 1, 2, 2, 2];
        expected_inner.iter().zip(expected_outer).enumerate().for_each(|(index, (inner, outer))|{
            assert_eq!(inner, get_one_d_array_inner_loop(index, num_cols, &u_array_test));
            assert_eq!(&outer, get_one_d_array_outer_loop(index, num_cols, &u_array_test));
        });
    }
    #[test]
    fn test_vasicek_mgf(){
        let y0=vec![0.9, 1.0, 1.1];
        let alpha=vec![0.2, 0.3, 0.2];
        let sigma=vec![0.2, 0.1, 0.2];
        let rho=vec![1.0, -0.4, 0.2, -0.4, 1.0, 0.3, 0.2, 0.3, 1.0];
        //this comes from c++ implementation https://github.com/phillyfan1138/Vasicek
        let expected=Complex::new(-19.9588, 2.253);
        let t=1.0;
        let expectation=compute_integral_expectation_long_run_one(&y0, &alpha, t);
        let variance=compute_integral_variance(&alpha, &sigma, &rho, t);
        let v_mgf=get_vasicek_mgf(expectation, variance);
        let result=v_mgf(&vec![Complex::new(1.0, 1.0), Complex::new(1.0, 1.0), Complex::new(1.0, 1.0)]);
        assert_abs_diff_eq!(result.re, expected.re, epsilon=0.0001);
        assert_abs_diff_eq!(result.im, expected.im, epsilon=0.0001);
    }
}