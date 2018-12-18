extern crate fang_oost;
extern crate num_complex;
extern crate rayon;
extern crate cf_functions;
extern crate rand;
extern crate utils;
use utils::vec_to_mat;
use utils::vasicek;
extern crate cf_dist_utils;
use self::num_complex::Complex;
use self::rayon::prelude::*;
#[macro_use]
extern crate serde_derive;
#[macro_use]
extern crate serde_json;
use std::io;
use std::io::prelude::*; //needed for write
use std::io::BufReader;
use std::io::BufRead;
use std::fs::File;
#[macro_use]
#[cfg(test)]
extern crate approx;

#[derive(Debug,Deserialize)]
struct Loan {
    balance:f64,
    pd:f64,
    lgd:f64,
    weight:Vec<f64>,
    #[serde(default = "default_num")]
    num:f64
}

fn default_num()->f64{
    1.0
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

//lambda needs to be made negative, the probability of lambda occurring is
// -qX since X is negative.
fn get_liquidity_risk_fn(
    lambda:f64,
    q:f64
)->impl Fn(&Complex<f64>)->Complex<f64>
{
    move |u:&Complex<f64>|u-((-u*lambda).exp()-1.0)*q//-u
}

struct HoldDiscreteCF {
    cf: Vec<Complex<f64> >,
    num_w: usize //num columns
}

impl HoldDiscreteCF {
    pub fn new(num_u: usize, num_w: usize) -> HoldDiscreteCF{
        HoldDiscreteCF{
            cf: vec![Complex::new(0.0, 0.0); num_u*num_w],
            num_w //num rows
        }
    }
    #[cfg(test)]
    pub fn get_cf(&self)->&Vec<Complex<f64>>{
        return &self.cf
    }
    pub fn process_loan<U>(
        &mut self, loan: &Loan, 
        u_domain: &[Complex<f64>],
        log_lpm_cf: U
    ) where U: Fn(&Complex<f64>, &Loan)->Complex<f64>+std::marker::Sync+std::marker::Send
    {
        let vec_of_cf_u:Vec<Complex<f64>>=u_domain
            .par_iter()
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
            *elem+=vec_of_cf_u[col_num]*loan.weight[row_num]*loan.num;
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
        //while "l" should be negative, note that the cir_mgf makes the "u" as negative 
        //since its derived from the CIR model which is a discounting model.
        //In general, implementations should make l negative when input into the 
        //lgd_cf
        cf_functions::cir_mgf(
            &(u*l), speed, long_run_average*speed, 
            sigma, t, x0
        )
    }   
}
#[cfg(test)]
fn test_mgf(u_weights:&[Complex<f64>])->Complex<f64>{
    u_weights.iter()
        .sum::<Complex<f64>>().exp()
}

#[cfg(test)]
fn gamma_mgf(variance:f64)->impl Fn(&[Complex<f64>])->Complex<f64>{
    let kappa=1.0/variance;//average is one
    move |u_weights:&[Complex<f64>]|->Complex<f64>{
        u_weights.iter().map(|u|{
            -(1.0-variance*u).ln()*kappa
        }).sum::<Complex<f64>>().exp()
    }
}

fn variance_liquidity(
    lambda:f64,
    q:f64,
    variance:f64,
    expectation:f64
)->f64{
    variance*(1.0+q*lambda).powi(2)+expectation*q*lambda.powi(2)
}
fn expectation_liquidity(
    lambda:f64,
    q:f64,
    expectation:f64
)->f64{
    expectation*(1.0+q*lambda)
}

fn variance_from_gamma(weights:&[f64], variances:&[f64])->f64{
    weights.iter().zip(variances).map(|(w, v){
        v*w.powi(2)
    }).sum()
}

fn generic_risk_contribution(
    pd:f64,
    expectation_l:f64,
    variance_l:f64,
    expectation_portfolio:f64,
    variance_portfolio:f64,
    variance_loan:f64,
    c:f64,
    rj:f64,
    balance:f64,
    lambda_0:f64,
    lambda:f64,
    q:f64
)->f64{
    let variance_liq=variance_liquidity(
        lambda, q, variance_portfolio, 
        expectation_portfolio
    );
    let coef=c/variance_liq.sqrt();
    pd*expectation_l*(1.0+q*lambda_0)+
        rj*balance*q*expectation_portfolio+
        coef*(
            pd*expectation_l*q*lambda_0.powi(2)+
            rj*balance*(lambda_0+lambda)*q*expectation_portfolio
        )+
        coef*(
            pd*expectation_l*variance_loan-
            pd*(variance_l+expectation_l.pow(2))
        )*(1.0+q*lambda_0).powi(2)+
        coef*(
            2.0*rj*balance*q*variance_portfolio+
            rj*balance*q.powi(2)*variance_portfolio*(lambda+lambda_0)
        )
}
fn scale_contributions(
    risk_measure:f64,
    expectation_liquid:f64,
    variance_liquid:f64
)->f64{
    (risk_measure-expectation_liquid)/variance_liquid
}
/*
fn risk_contribution_existing_loan(
    loan:&Loan, gamma_variances:&[f64], risk_measure:f64,
    variance_l:f64,
    expectation_portfolio:f64, 
    variance_portfolio:f64
)->f64{
    generic_risk_contribution(
        loan.pd, loan.lgd*loan.balance, 
        variance_l, expectation_portfolio, 
        variance_portfolio, 
        variance_from_gamma(&loan.weight, gamma_variances),
        scale_contributions(
            risk_measure, 
        )
    )
}*/

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
        num_u, num_w
    );

    let u_domain:Vec<Complex<f64>>=fang_oost::get_u_domain(
        num_u, x_min, x_max
    ).collect();

    let f = File::open(args[2].as_str())?;
    let file = BufReader::new(&f);
    for line in file.lines() {
        let loan: Loan = serde_json::from_str(&line?)?;
        discrete_cf.process_loan(&loan, &u_domain, &log_lpm_cf);
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
    if args.len()>3 {
        let x_domain:Vec<f64>=fang_oost::get_x_domain(1024, x_min, x_max).collect();
        let density:Vec<f64>=fang_oost::get_density(
            x_min, x_max, 
            fang_oost::get_x_domain(1024, x_min, x_max), 
            &final_cf
        ).collect();
        let json_results=json!({"x":x_domain, "density":density});
        let mut file_w = File::create(args[3].as_str())?;
        file_w.write_all(json_results.to_string().as_bytes())?;
    }
    

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
    let variance_portfolio=cf_dist_utils::get_variance_discrete_cf(
        x_min, x_max, &final_cf
    );
    let expectation_portfolio=cf_dist_utils::get_expectation_discrete_cf(
        x_min, x_max, &final_cf
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
            256, 3
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
            256, 3
        );
        let loan=Loan{
            pd:0.05,
            lgd:0.5,
            balance:1000.0,
            weight:vec![0.5, 0.5, 0.5],
            num:1.0
        };
        let log_lpm_cf=|_u:&Complex<f64>, _loan:&Loan|{
            Complex::new(1.0, 0.0)
        };
        let u_domain:Vec<Complex<f64>>=fang_oost::get_u_domain(
            256, 0.0, 1.0
        ).collect();
        discrete_cf.process_loan(&loan, &u_domain, &log_lpm_cf);
        let cf=discrete_cf.get_cf();
        assert_eq!(cf.len(), 256*3);
        cf.iter().for_each(|cf_el|{
            assert_eq!(cf_el, &Complex::new(0.5 as f64, 0.0 as f64));
        });
        
    }
    #[test]
    fn test_process_loans_with_final(){
        let mut discrete_cf=HoldDiscreteCF::new(
            256, 3
        );
        let loan=Loan{
            pd:0.05,
            lgd:0.5,
            balance:1000.0,
            weight:vec![0.5, 0.5, 0.5],
            num:1.0
        };
        let u_domain:Vec<Complex<f64>>=fang_oost::get_u_domain(
            256, 0.0, 1.0
        ).collect();
        let log_lpm_cf=|_u:&Complex<f64>, _loan:&Loan|{
            Complex::new(1.0, 0.0)
        };
        discrete_cf.process_loan(&loan, &u_domain, &log_lpm_cf);
        let final_cf:Vec<Complex<f64>>=discrete_cf.get_full_cf(&test_mgf);
    
        assert_eq!(final_cf.len(), 256);
        final_cf.iter().for_each(|cf_el|{
            assert_eq!(cf_el, &Complex::new(1.5 as f64, 0.0 as f64).exp());
        });
    }
    #[test]
    fn test_actually_get_density(){
        let x_min=-6000.0;
        let x_max=0.0;
        let mut discrete_cf=HoldDiscreteCF::new(
            256, 1
        );
        let lambda=1000.0;
        let q=0.0001;
        let liquid_fn=get_liquidity_risk_fn(lambda, q);

        let t=1.0;
        let alpha_l=0.2;
        let b_l=1.0;
        let sig_l=0.2;
        let lgd_fn=get_lgd_cf_fn(alpha_l, b_l, sig_l, t, b_l);//assumption is that it starts at the lgd mean...
        let u_domain:Vec<Complex<f64>>=fang_oost::get_u_domain(
            256, x_min, x_max
        ).collect();
        let log_lpm_cf=get_log_lpm_cf(&lgd_fn, &liquid_fn);

        let loan=Loan{
            pd:0.05,
            lgd:0.5,
            balance:1.0,
            weight:vec![1.0],
            num:10000.0
        };
        discrete_cf.process_loan(&loan, &u_domain, &log_lpm_cf);
        let y0=vec![1.0];
        let alpha=vec![0.3];
        let sigma=vec![0.3];
        let rho=vec![1.0];
        let t=1.0;
        let expectation=vasicek::compute_integral_expectation_long_run_one(
            &y0, &alpha, t
        );
        let variance=vasicek::compute_integral_variance(
            &alpha, &sigma, 
            &rho, t
        );

        let v_mgf=vasicek::get_vasicek_mgf(expectation, variance);
        
        let final_cf:Vec<Complex<f64>>=discrete_cf.get_full_cf(&v_mgf);

        assert_eq!(final_cf.len(), 256);
        let max_iterations=100;
        let tolerance=0.0001;
        let (
            es, 
            var
        )=cf_dist_utils::get_expected_shortfall_and_value_at_risk_discrete_cf(
            0.01, 
            x_min,
            x_max,
            max_iterations,
            tolerance,
            &final_cf
        );
        println!("this is es: {}", es);
        println!("this is var: {}", var);
        assert!(es>var);
    }
    #[test]
    fn test_gamma_cf(){
        let kappa=2.0;
        let theta=0.5;
        let u=Complex::new(0.5, 0.5);
        let cf=gamma_mgf(theta);
        let result=cf(&vec![u]);
        let expected=(1.0-u*theta).powf(-kappa);
        assert_eq!(result, expected);
    }
    #[test]
    fn test_compare_expected_value(){
        let balance=1.0;
        let pd=0.05;
        let lgd=0.5;
        let num_loans=10000.0;
        let lambda=1000.0; //loss in the event of a liquidity crisis
        let q=0.01/(num_loans*pd*lgd*balance);
        let expectation=-pd*lgd*balance*(1.0+lambda*q)*num_loans;
        let x_min=(expectation-lambda)*3.0;
        let x_max=0.0;
        let num_u:usize=1024;
        let mut discrete_cf=HoldDiscreteCF::new(
            num_u, 1
        );
       
        let liquid_fn=get_liquidity_risk_fn(lambda, q);

        //the exponent is negative because l represents a loss
        let lgd_fn=|u:&Complex<f64>, l:f64|(-u*l).exp();
        
        let u_domain:Vec<Complex<f64>>=fang_oost::get_u_domain(
            num_u, x_min, x_max
        ).collect();
        let log_lpm_cf=get_log_lpm_cf(&lgd_fn, &liquid_fn);
        
        let loan=Loan{
            pd,
            lgd,
            balance,
            weight:vec![1.0],
            num:num_loans//homogenous
        };
        discrete_cf.process_loan(&loan, &u_domain, &log_lpm_cf);
        let v=0.3;
        let v_mgf=gamma_mgf(v);        
        let final_cf:Vec<Complex<f64>>=discrete_cf.get_full_cf(&v_mgf);
        assert_eq!(final_cf.len(), num_u);
        let expectation_approx=cf_dist_utils::get_expectation_discrete_cf(x_min, x_max, &final_cf);
        
        assert_abs_diff_eq!(expectation_approx, expectation, epsilon=0.00001);
    }
}