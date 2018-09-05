extern crate num_complex;
use self::num_complex::Complex;
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
fn get_two_d_array<T:Copy>(
    row_num:usize,
    col_num:usize,
    num_cols:usize,
    array:&[T]
)->T{
    array[row_num*num_cols+col_num]
}
fn get_one_d_array_outer_loop<T:Copy>(
    index:usize, 
    num_cols:usize,
    array:&[T]
)->T {
    array[index%num_cols]
}
fn get_one_d_array_inner_loop<T:Copy>(
    index:usize, 
    num_cols:usize,
    array:&[T]
)->T {
    array[index/num_cols] //rounds down
}
pub fn compute_integral_variance(
    alpha:&[f64],
    sigma:&[f64],
    rho:&[f64],
    t:f64
)->Vec<f64>{
    let num_cols=alpha.len();
    alpha.iter().zip(sigma).enumerate().flat_map(|(i, (&alpha_elem_i, &sigma_elem_i))|{
        let a_i=help_compute_moments(alpha_elem_i, t);
        alpha.iter().zip(sigma).enumerate().map(move |(j, (&alpha_elem_j, &sigma_elem_j))|{
            let a_j=help_compute_moments(alpha_elem_j, t);
            cross_multiply(
                get_two_d_array(i, j, num_cols, rho),
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
)->impl Fn(&Vec<Complex<f64>>)->Complex<f64>{
    let num_cols=expectation.len();
    move |u_vec|{
        expectation.iter().zip(u_vec).map(|(exp_increment, u_increment)|{
            exp_increment*u_increment
        }).sum::<Complex<f64>>()+variance.iter().enumerate().map(|(index, var_increment)|{
            var_increment
                *get_one_d_array_inner_loop(index, num_cols, u_vec)
                *get_one_d_array_outer_loop(index, num_cols, u_vec)
        }).sum::<Complex<f64>>()*0.5
    }
}

pub fn get_vasicek_mgf(
    expectation:Vec<f64>,
    variance:Vec<f64>
)->impl Fn(&Vec<Complex<f64>>)->Complex<f64>{
    let log_vasicek_mgf=get_log_vasicek_mgf(expectation, variance);
    move |u_vec|log_vasicek_mgf(u_vec).exp()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_get_two_d_array(){
        let arr=vec![1, 2, 3, 4, 5, 6];
        //let result=get_two_d_array(1, 0, &arr)
        let num_cols=3;
        let row_index_1=1;
        let col_index_1=0;
        let row_index_2=0;
        let col_index_2=1;
        assert_eq!(get_two_d_array(row_index_1, col_index_1, num_cols, &arr), 4);
        assert_eq!(get_two_d_array(row_index_2, col_index_2, num_cols, &arr), 2);
    }
    #[test]
    fn test_get_one_d_array_outer(){
        //let arr=vec![1, 2, 3, 4, 5, 6];
        let arr=vec![1, 2, 3];
        //let result=get_two_d_array(1, 0, &arr)
        let num_cols=3;
        assert_eq!(get_one_d_array_outer_loop(4, num_cols, &arr), 2);
        assert_eq!(get_one_d_array_outer_loop(0, num_cols, &arr), 1);
    }
    #[test]
    fn test_get_one_d_array_inner(){
        //let arr=vec![1, 2, 3, 4, 5, 6];
        let arr=vec![1, 2, 3];
        //let result=get_two_d_array(1, 0, &arr)
        let num_cols=3;
        assert_eq!(get_one_d_array_inner_loop(4, num_cols, &arr), 2);
        assert_eq!(get_one_d_array_inner_loop(2, num_cols, &arr), 1);
        assert_eq!(get_one_d_array_inner_loop(0, num_cols, &arr), 1);
    }
    #[test]
    fn test_compute_expectation_long_run(){
        let y0=vec![0.5];
        let alpha=vec![0.3];
        let beta=vec![0.5];
        let t=1.0;
        //let single_expectation=
        assert_eq!(*compute_expectation(&y0, &alpha, &beta, t).first().unwrap(), beta[0]);
    }
}