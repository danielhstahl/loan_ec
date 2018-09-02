extern crate fang_oost;
extern crate num_complex;
extern crate rayon;
extern crate cf_functions;
use self::num_complex::Complex;
use self::rayon::prelude::*;

extern crate serde_json;
#[macro_use]
extern crate serde_derive;

use std::env;
use std::io;

#[derive(Serialize, Deserialize)]
struct Loan {
    balance:f64,
    weight:Vec<f64>,
    pd:f64,
    lgd:f64
}
#[derive(Serialize, Deserialize)]
struct Parameters {
    lambda:f64,
    q:f64,
    alpha_l:f64,
    b_l:f64,
    sig_l:f64,
    t:f64,
    u_steps:usize,
    num_send:usize,
    x_min:f64,
    x_max:f64
}


fn get_liquidity_risk_fn(
    lambda:f64,
    q:f64
)->impl Fn(&Complex<f64>)->Complex<f64>
{
    move |u:&Complex<f64>|-((-u*lambda).exp()-1.0)*q-u
}

fn get_full_exponent<T, U>(
    x_min:f64,
    x_max:f64,
    num_u:usize,
    get_liquidity:T, 
    log_lpm_cf:U 
)->impl Fn( &[Loan])->Vec<Vec<Complex<f64> > > 
where 
    T:Fn(&Complex<f64>)->Complex<f64>+std::marker::Sync+std::marker::Send,
    U: Fn(&Complex<f64>, &[Loan])->Complex<f64>+std::marker::Sync+std::marker::Send
{
    let du=fang_oost::compute_du(x_min, x_max);
    move |loans:&[Loan]|{
        (0..num_u)
            .into_par_iter()
            .map(|index|{
                log_lpm_cf(
                    &get_liquidity(
                        &fang_oost::get_complex_u(fang_oost::get_u(du, index))
                    ),
                    loans
                )
            }).collect()//array of size num_u by num_systemic_factors
    }
}

fn log_lpm_cf<T, U, V, W>(
    num_systemic_factors:usize,
    lgd_cf:T,
    get_l:U,
    get_pd:V,
    get_w:W 
)->impl Fn(&Complex<f64>, &[Loan])->Vec<Complex<f64> >
where
    T:Fn(&Complex<f64>, f64)->Complex<f64> +std::marker::Sync+std::marker::Send,
    U:Fn(&Loan)->f64 +std::marker::Sync+std::marker::Send,
    V:Fn(&Loan)->f64 +std::marker::Sync+std::marker::Send,
    W: Fn(&Loan, usize)->f64+std::marker::Sync+std::marker::Send
{ 
    move |u:&Complex<f64>, loans:&[Loan]|{
        (0..num_systemic_factors).map(|index|{
            loans.iter().fold(Complex::new(0.0, 0.0), |accum, loan|{
                (lgd_cf(u, get_l(loan))-1.0)*get_pd(loan)*get_w(loan, index)
            })
        }).collect()//array of size num_systemic_factors
    }
}
fn get_lgd_cf_fn(
    speed:f64,
    long_run_average:f64,
    sigma:f64,
    t:f64,
    x0:f64
)->impl Fn(&Complex<f64>, f64)->Complex<f64>{
    |u:&Complex<f64>, l:f64|{
        cf_functions::cir_mgf(
            &(u*l), speed, long_run_average*speed, 
            sigma, t, x0
        )
    }   
}

fn main()-> Result<(), io::Error> {
    let args: Vec<String> = env::args().collect();
    let parameters:Parameters=serde_json::from_str(&args[1])?;
    let loans:Vec<Loan>=serde_json::from_str(&args[2])?;
    let num_systemic_factors=loans.first().unwrap().weight.len();
    let Parameters{
        lambda,
        q,
        alpha_l,
        b_l,
        sig_l,
        t,
        u_steps,
        num_send,
        x_min,
        x_max
    }=parameters;

    let liquid_fn=get_liquidity_risk_fn(lambda, q);
    let lgd_fn=get_lgd_cf_fn(alpha_l, b_l, sig_l, t, b_l);//assumption is that it starts at the lgd mean...
    
    let log_loan_cf_fn=log_lpm_cf(
        num_systemic_factors, &lgd_fn, 
        &|&loan|loan.balance*loan.lgd,
        &|&loan|loan.pd,
        &|&loan, index|loan.weight[index]
    );
    let full_exponent=get_full_exponent(x_min, x_max, u_steps, &liquid_fn, &log_loan_cf_fn);
    let complex_exponent=full_exponent(&loans); //and we have a num_u by num_systemic factor vector of vectors
    println!("num u {}:", complex_exponent.len());
    println!("num systemic factor {}:", complex_exponent.first().unwrap().len());
}
