extern crate fang_oost;
extern crate num_complex;
extern crate rayon;
extern crate cf_functions;
extern crate rand;
extern crate utils;
use utils::vec_to_mat;
use utils::vasicek;
extern crate cf_dist_utils;
//extern crate csv;
use self::num_complex::Complex;
use self::rayon::prelude::*;
#[macro_use]
extern crate serde_derive;
extern crate serde_json;
use std::io;

use std::io::BufReader;
use std::io::BufRead;
use std::fs::File;

#[derive(Debug,Deserialize)]
struct Loan {
    balance:f64,
    pd:f64,
    lgd:f64,
    weight:Vec<f64>
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
    //#[cfg(test)]
    pub fn get_cf(&self)->&Vec<Complex<f64>>{
        return &self.cf
    }
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
        let num_w=self.num_w;
        self.cf.par_iter_mut().enumerate().for_each(|(index, elem)|{
            let row_num=vec_to_mat::get_row_from_index(index, num_w);
            let col_num=vec_to_mat::get_col_from_index(index, num_w);
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

fn get_log_lpm_cf<T, U>(
    lgd_cf:T,
    liquidity_cf:U
)-> impl Fn(&Complex<f64>, &Loan)->Complex<f64>
    where T: Fn(&Complex<f64>, f64)->Complex<f64>,
          U: Fn(&Complex<f64>)->Complex<f64>
{
    move |u:&Complex<f64>, loan:&Loan|{
        (lgd_cf(&liquidity_cf(u), loan.lgd*loan.balance)-1.0)*loan.pd
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

fn test_mgf(u_weights:&[Complex<f64>])->Complex<f64>{
    u_weights.iter()
        .sum::<Complex<f64>>().exp()
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
    
    let log_lpm_cf=get_log_lpm_cf(&lgd_fn, &liquid_fn);

    let mut discrete_cf=HoldDiscreteCF::new(
        num_u, num_w, 
        x_min, x_max
    );
    //let f = File::open("./data/loans.json")?;
    let f = File::open(args[2].as_str())?;
    let file = BufReader::new(&f);
    for line in file.lines() {
        let loan: Loan = serde_json::from_str(&line?)?;
        discrete_cf.process_loan(&loan, &log_lpm_cf);
    }  
    
    let expectation=vasicek::compute_integral_expectation_long_run_one(
        &y0, &alpha, t
    );
    let variance=vasicek::compute_integral_variance(
        &alpha, &sigma, 
        &rho, t
    );

    /*for v_inc in variance.iter(){
        println!("this is variance {}", v_inc);
    }
    for v_inc in expectation.iter(){
        println!("this is expectation {}", v_inc);
    }*/

    let v_mgf=vasicek::get_vasicek_mgf(expectation, variance);


    for cf_inc in discrete_cf.get_cf(){
        println!("this is cf before vas {}", cf_inc);
    }

    

    let final_cf:Vec<Complex<f64>>=discrete_cf.get_full_cf(&test_mgf);
    //println!("{}", final_cf[0]);
    /*for cf_inc in final_cf.iter(){
        println!("{}", cf_inc);
    }*/
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
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn construct_hold_discrete_cf(){
        let discrete_cf=HoldDiscreteCF::new(
            256, 3, 0.0, 1.0
        );
        let cf=discrete_cf.get_cf();
        assert_eq!(cf.len(), 256*3);
        assert_eq!(cf[0], Complex::new(0.0, 0.0)); //first three should be the same "u"
        assert_eq!(cf[1], Complex::new(0.0, 0.0));
        assert_eq!(cf[2], Complex::new(0.0, 0.0));
    }
    #[test]
    fn test_process_loan(){
        let mut discrete_cf=HoldDiscreteCF::new(
            256, 3, 0.0, 1.0
        );
        let loan=Loan{
            pd:0.05,
            lgd:0.5,
            balance:1000.0,
            weight:vec![0.5, 0.5, 0.5]
        };
        let log_lpm_cf=|_u:&Complex<f64>, _loan:&Loan|{
            Complex::new(1.0, 0.0)
        };
        discrete_cf.process_loan(&loan, &log_lpm_cf);
        let cf=discrete_cf.get_cf();
        assert_eq!(cf.len(), 256*3);
        cf.iter().for_each(|cf_el|{
            assert_eq!(cf_el, &Complex::new(0.5 as f64, 0.0 as f64));
        });
        
    }
    #[test]
    fn test_process_loans_with_final(){
        let mut discrete_cf=HoldDiscreteCF::new(
            256, 3, 0.0, 1.0
        );
        let loan=Loan{
            pd:0.05,
            lgd:0.5,
            balance:1000.0,
            weight:vec![0.5, 0.5, 0.5]
        };
        let log_lpm_cf=|_u:&Complex<f64>, _loan:&Loan|{
            Complex::new(1.0, 0.0)
        };
        discrete_cf.process_loan(&loan, &log_lpm_cf);
        let final_cf:Vec<Complex<f64>>=discrete_cf.get_full_cf(&test_mgf);
    
        assert_eq!(final_cf.len(), 256);
        final_cf.iter().for_each(|cf_el|{
            assert_eq!(cf_el, &Complex::new(1.5 as f64, 0.0 as f64).exp());
        });
        
    }
}