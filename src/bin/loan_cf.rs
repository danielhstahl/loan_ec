extern crate fang_oost;
extern crate num_complex;
extern crate rayon;
extern crate cf_functions;
extern crate rand;
extern crate vasicek;
extern crate cf_dist_utils;
extern crate csv;
use self::num_complex::Complex;
use self::rayon::prelude::*;
#[macro_use]
extern crate serde_derive;
#[macro_use]
extern crate serde_json;


use std::fs::File;
use std::io::prelude::*;

use std::io;



#[derive(Debug,Deserialize)]
struct Loan {
    balance:f64,
    pd:f64,
    lgd:f64,
    //#[serde(flatten)]
    weight: Vec<f64>
}
#[derive(Debug,Deserialize)]
#[serde(rename_all = "camelCase")]
struct Parameters {
    lambda:f64,
    q:f64,
    alpha_l:f64,
    b_l:f64,
    sig_l:f64,
    t:f64,
    num_u:usize,
    x_min:f64,
    x_max:f64,
    num_x:Option<usize>,
    alpha:Vec<f64>,
    sigma:Vec<f64>,
    rho:Vec<f64>,
    y0:Vec<f64>
}


fn get_liquidity_risk_fn(
    lambda:f64,
    q:f64
)->impl Fn(&Complex<f64>)->Complex<f64>
{
    move |u:&Complex<f64>|-((-u*lambda).exp()-1.0)*q-u
}

struct HoldDiscreteCF {
    cf: Vec<Complex<f64> >,
    x_min: f64,
    x_max: f64,
    num_u: usize, //num rows
    num_w: usize //num columns
}

fn get_col_from_index(index:usize, num_row:usize)->usize {
    index%num_row
}
fn get_row_from_index(index:usize, num_col:usize)->usize {
    index/num_col
}
/*fn get_element_at<T>(arbitrary_vec:&[T], num_rows:usize, row_num:usize, col_num:usize)->T{
    arbitrary_vec[col_num*num_rows+row_num]
}*/

impl HoldDiscreteCF {
    pub fn new(num_u: usize, num_w: usize, x_min:f64, x_max:f64) -> HoldDiscreteCF{
        HoldDiscreteCF{
            cf: vec![Complex::new(0.0, 0.0); num_u*num_w],
            num_u, //num columns
            num_w, //num rows
            x_min,
            x_max
        }
    }
    /*pub fn get_element_at(&self, row_num: usize, col_num: usize)->&Complex<f64>{
        get_element_at(&self.cf, self.num_u, row_num, col_num)
    }*/
    pub fn process_loan<U>(
        &mut self, loan: 
        &Loan, 
        log_lpm_cf:U
    ) where U: Fn(&Complex<f64>, &Loan)->Complex<f64>+std::marker::Sync+std::marker::Send
    {
        let vec_of_cf_u:Vec<Complex<f64>>=fang_oost::get_u_domain(
            self.num_u, self.x_min, self.x_max
        )
            .map(|u|{
                log_lpm_cf(
                    &u, 
                    loan
                )
            }).collect(); 
        let num_u=self.num_u;
        let num_w=self.num_w;
        self.cf.par_iter_mut().enumerate().for_each(|(index, elem)|{
        //for (index, elem) in self.cf.par_iter_mut().enumerate(){
            let row_num=get_row_from_index(index, num_u);
            let col_num=get_col_from_index(index, num_w);
            *elem+=vec_of_cf_u[col_num]*loan.weight[row_num];
        });
    }
    pub fn get_full_cf<U>(&self, mgf:U)->Vec<Complex<f64>>
    where U: Fn(&[Complex<f64>])->Complex<f64>+std::marker::Sync+std::marker::Send
    {
        self.cf.par_chunks(self.num_w)
            .map(mgf).collect()
    }
}

fn get_log_lpm_cf<T>(
    lgd_cf:T
)-> impl Fn(&Complex<f64>, &Loan)->Complex<f64>
    where T: Fn(&Complex<f64>, f64)->Complex<f64>
{
    move |u:&Complex<f64>, loan:&Loan|{
        (lgd_cf(u, loan.lgd*loan.balance)-1.0)*loan.pd
    }
}

fn get_lgd_cf_fn(
    speed:f64,
    long_run_average:f64,
    sigma:f64,
    t:f64,
    x0:f64
)->impl Fn(&Complex<f64>, f64)->Complex<f64>{
    move |u:&Complex<f64>, l:f64|{
        cf_functions::cir_mgf(
            &(u*l), speed, long_run_average*speed, 
            sigma, t, x0
        )
    }   
}


fn main()-> Result<(), io::Error> {
    let args: Vec<String> = std::env::args().collect();
    let Parameters{
        lambda, q, alpha_l, 
        b_l, sig_l, t, num_u, 
        x_min, x_max, alpha, 
        sigma, rho, y0, ..
    }= serde_json::from_str(args[1].as_str())?;
    let num_w=alpha.len();
    let liquid_fn=get_liquidity_risk_fn(lambda, q);
    let lgd_fn=get_lgd_cf_fn(alpha_l, b_l, sig_l, t, b_l);//assumption is that it starts at the lgd mean...
    
    let log_lpm_cf=get_log_lpm_cf(&lgd_fn);

    let mut discrete_cf=HoldDiscreteCF::new(
        num_u, num_w, 
        x_min, x_max
    );


    let mut rdr = csv::Reader::from_reader(io::stdin());
    for result in rdr.deserialize() {
        // Notice that we need to provide a type hint for automatic
        // deserialization.
        let loan: Loan = result?;
        discrete_cf.process_loan(&loan, &log_lpm_cf);
        //println!("{:?}", record);
    }

    
    let expectation=vasicek::compute_integral_expectation_long_run_one(
        &y0, &alpha, t
    );
    let variance=vasicek::compute_integral_variance(
        &alpha, &sigma, 
        &rho, t
    );
    let v_mgf=vasicek::get_vasicek_mgf(expectation, variance);

    let final_cf:Vec<Complex<f64>>=discrete_cf.get_full_cf(&v_mgf);

    /*let x_domain:Vec<f64>=fang_oost::get_x_domain(
        num_x, x_min, x_max
    ).collect();*/
    //let density:Vec<f64>=fang_oost::get_density(x_min, x_max, fang_oost::get_x_domain(x_steps, x_min, x_max), &final_cf).collect();

    let max_iterations=100;
    let tolerance=0.0001;
    let (es, var)=cf_dist_utils::get_expected_shortfall_and_value_at_risk_discrete_cf(
        0.01, 
        x_min,
        x_max,
        max_iterations,
        tolerance,
        &final_cf
    );
    println!("This is ES: {}", es);
    println!("This is VaR: {}", var);
    Ok(())
}
