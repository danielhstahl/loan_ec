extern crate fang_oost;
extern crate num_complex;
extern crate rayon;
extern crate cf_functions;
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
extern crate probability;
use probability::prelude::*;
#[macro_use]
#[cfg(test)]
extern crate approx;

#[derive(Debug,Deserialize)]
#[serde(rename_all = "camelCase")]
struct Parameters {
    lambda:f64,
    q:f64,
    num_u:usize,
    x_min:f64,
    x_max:f64,
    num_x:Option<usize>,
    r_squared:Vec<f64>
}

//for the bivariate cdf...see https://apps.dtic.mil/dtic/tr/fulltext/u2/a125033.pdf
fn mean_h(h:f64, rho:f64, normal:&Gaussian)->f64{
    -normal.density(h)*rho/normal.distribution(h)
}
fn var_h(h:f64, rho:f64, mean:f64)->f64{
    1.0+rho*h*mean-mean.powi(2)
}

//needed to convert from variance/covariance in Merton to Credit Risk Plus
fn biv_gaussian_orig(x1:f64, x2:f64, rho:f64, normal:&Gaussian)->f64{
    let mean=mean_h(x1, rho, &normal);
    let variance=var_h(x1, rho, mean);
    normal.distribution(x1)
        *normal.distribution((x2-mean)/variance.sqrt())
}

fn biv_gaussian(x1:f64, x2:f64, rho:f64, normal:&Gaussian)->f64{
    let c=if x1>x2{x1}else{x2};
    let d=if x1>x2{x2}else{x1};
    if c<0.0{
        biv_gaussian_orig(c, d, rho, normal)
    }
    else{
        normal.distribution(d)-biv_gaussian_orig(-c, d, -rho, normal)
    }
}

fn cov_merton(p:f64, rho:f64)->f64{
    let normal=Gaussian::new(0.0, 1.0);
    let x=normal.inverse(p);
    biv_gaussian(x, x, rho, &normal)
}
//converts to variance
fn get_systemic_variance(p:f64, rho:f64)->f64{
    cov_merton(p, rho)/p.powi(2)-1.0
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


fn main()-> Result<(), io::Error> {
    let args: Vec<String> = std::env::args().collect();
    let Parameters{
        lambda, q,  
        num_u, x_min, 
        x_max, r_squared, 
        ..
    }= serde_json::from_str(args[1].as_str())?;
    let num_w=r_squared.len();
    let p=0.05;//just for tests
    let systemic_variance=r_squared.iter().map(|r|{
        get_systemic_variance(p, r.sqrt())//sqrt since given r-squared
    }).collect::<Vec<_>>();
    
    systemic_variance.iter().for_each(|v|{
        println!("this is var: {}", v);
    });
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
    //let temp_v=vec![0.5; 20];
    let v_mgf=gamma_mgf(systemic_variance); 
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
    #[test]
    fn cov_merton_compare_r_1(){
        let p=0.05;
        let rho=0.5;
        let result=cov_merton(p, rho);
        assert_abs_diff_eq!(
            result, 
            0.01218943, 
            epsilon=0.001
        );
    }
    #[test]
    fn biv_gaussian_compare_r_1(){
        let rho=0.7;
        let k=1.0;
        let h=0.2;
        let normal=Gaussian::new(0.0, 1.0);
        let result=biv_gaussian(h, k, rho, &normal);
        assert_abs_diff_eq!(
            result, 
            0.55818, 
            epsilon=0.00001
        );
    }
    #[test]
    fn biv_gaussian_compare_r_2(){
        let rho=-0.7;
        let k=-1.0;
        let h=0.2;
        let normal=Gaussian::new(0.0, 1.0);
        let result=biv_gaussian_orig(k, h, rho, &normal);
        assert_abs_diff_eq!(
            result, 
            0.02108,
            epsilon=0.00001
        );
    }
    #[test]
    fn mean_compare_r_1(){
        let rho=-0.7;
        let normal=Gaussian::new(0.0, 1.0);
        let h=-1.0;
        let result=mean_h(h, rho, &normal);
        assert_abs_diff_eq!(
            result, 
            1.0676, 
            epsilon=0.00001
        );
    }
    #[test]
    fn var_compare_r_1(){
        let rho=-0.7;
        let h=-1.0;
        let normal=Gaussian::new(0.0, 1.0);
        let mean=mean_h(h, rho, &normal);
        let var=var_h(h, rho,  mean);
        assert_abs_diff_eq!(
            var.sqrt(), 
            0.77946,
            epsilon=0.00001
        );
    }
}