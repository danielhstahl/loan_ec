extern crate fang_oost;
extern crate num_complex;
extern crate rayon;
extern crate cf_functions;
extern crate rand;
extern crate utils;
use utils::loan_ec;
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


fn gamma_mgf(variance:Vec<f64>)->
   impl Fn(&[Complex<f64>])->Complex<f64>
{
    move |u_weights:&[Complex<f64>]|->Complex<f64>{
        u_weights.iter().zip(&variance).map(|(u, v)|{
            -(1.0-v*u).ln()/v
        }).sum::<Complex<f64>>().exp()
    }
}
/*
fn risk_contribution_existing_loan(
    loan:&Loan, gamma_variances:&[f64], risk_measure:f64,
    variance_l:f64,
    expectation_portfolio:f64, 
    variance_portfolio:f64,
    lambda:f64, q:f64
)->f64{
    let expectation_liquid=risk_contributions::expectation_liquidity(
        lambda, q, expectation_portfolio
    );
    let variance_liquid=risk_contributions::variance_liquidity(
        lambda, q, expectation_portfolio, variance_portfolio
    );
    let rj=0.0;
    risk_contributions::generic_risk_contribution(
        loan.pd, loan.lgd*loan.balance, 
        variance_l, expectation_portfolio, 
        variance_portfolio, 
        risk_contributions::variance_from_independence(&loan.weight, gamma_variances),
        risk_contributions::scale_contributions(
            risk_measure, expectation_liquid, variance_liquid
        ),
        rj,
        loan.balance,
        lambda, 
        lambda,
        q
    )
}*/

fn main()-> Result<(), io::Error> {
    let args: Vec<String> = std::env::args().collect();
    let Parameters{
        lambda, q,  
         num_u, 
        x_min, x_max, alpha, 
        ..
    }= serde_json::from_str(args[1].as_str())?;
    let num_w=alpha.len();
    let liquid_fn=loan_ec::get_liquidity_risk_fn(lambda, q);
    let lgd_fn=|u:&Complex<f64>, l:f64, lgd_v:f64|{
        if lgd_v>0.0{
            cf_functions::gamma_cf(
                &(-u*l), 1.0/lgd_v, lgd_v
            )
        }
        else{
            (-u*l).exp()
        }
    };
    let log_lpm_cf=loan_ec::get_log_lpm_cf(&lgd_fn, &liquid_fn);

    let mut discrete_cf=loan_ec::EconomicCapitalAttributes::new(
        num_u, num_w
    );

    let u_domain:Vec<Complex<f64>>=fang_oost::get_u_domain(
        num_u, x_min, x_max
    ).collect();

    let f = File::open(args[2].as_str())?;
    let file = BufReader::new(&f);
    for line in file.lines() {
        let loan: loan_ec::Loan = serde_json::from_str(&line?)?;
        discrete_cf.process_loan(&loan, &u_domain, &log_lpm_cf);
    }  
    //TODO!! get variance from R^2
    let temp_v=vec![0.5; 20];
    let v_mgf=gamma_mgf(temp_v); 
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
    
    println!("This is ES: {}", es);
    println!("This is VaR: {}", var);
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_gamma_cf(){
        let kappa=2.0;
        //let theta=0.5;
        let u=Complex::new(0.5, 0.5);
        let theta=0.5;
        let cf=gamma_mgf(vec![theta]);
        let result=cf(&vec![u]);
        let expected=(1.0-u*theta).powf(-kappa);
        assert_eq!(result, expected);
    }
}