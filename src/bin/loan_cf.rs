extern crate fang_oost;
extern crate num_complex;
extern crate rayon;
extern crate cf_functions;
extern crate rand;
extern crate vasicek;
extern crate cf_dist_utils;
use rand::prelude::*;
use self::num_complex::Complex;
use self::rayon::prelude::*;

#[macro_use]
extern crate serde_json;


use std::fs::File;
use std::io::prelude::*;

use std::io;




struct Loan {
    balance:f64,
    weight:Vec<f64>,
    pd:f64,
    lgd:f64
}

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
    num_systemic:usize,
    get_liquidity:T, 
    log_lpm_cf:U 
)->impl Fn( &[Loan])->Vec<Complex<f64> > 
where 
    T:Fn(&Complex<f64>)->Complex<f64>+std::marker::Sync+std::marker::Send,
    U: Fn(&Complex<f64>, &[Loan], usize)->Complex<f64>+std::marker::Sync+std::marker::Send
{
    move |loans:&[Loan]|{
        fang_oost::get_u_domain(num_u, x_min, x_max)
            .flat_map(|u|{
                (0..num_systemic).map(|systemic_index|{
                    log_lpm_cf(
                        &get_liquidity(
                            &u
                        ),
                        loans,
                        systemic_index
                    )
                }).collect::<Vec<_>>()
            }).collect()//size is num_systemic*num_u
    }
}

fn log_lpm_cf<T, U, V, W>(
    lgd_cf:T,
    get_l:U,
    get_pd:V,
    get_w:W
)->impl Fn(&Complex<f64>, &[Loan], usize)-> Complex<f64>
where
    T:Fn(&Complex<f64>, f64)->Complex<f64> +std::marker::Sync+std::marker::Send,
    U:Fn(&Loan)->f64 +std::marker::Sync+std::marker::Send,
    V:Fn(&Loan)->f64 +std::marker::Sync+std::marker::Send,
    W: Fn(&Loan, usize)->f64+std::marker::Sync+std::marker::Send
{ 
    move |u:&Complex<f64>, loans:&[Loan], index:usize|{
        loans.iter().map(|loan|{
            (lgd_cf(u, get_l(loan))-1.0)*get_pd(loan)*get_w(loan, index)
        }).sum()
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
fn get_rough_total_exposure(
    min_loan_size:f64,
    max_loan_size:f64,
    num_loans:usize
)->f64{
    (num_loans as f64)*(min_loan_size+0.5*(max_loan_size-min_loan_size))
}
fn get_parameters(
    batch_size:usize,
    num_batches:usize,
    min_loan_size:f64,
    max_loan_size:f64,
    max_possible_loss:f64
)->Parameters{
    let exposure=get_rough_total_exposure(
        min_loan_size,
        max_loan_size,
        num_batches*batch_size
    );
    let lambda=0.7*exposure*max_possible_loss; //need to be less than max loss in dollars
    Parameters{
        lambda,
        q:0.2/lambda,
        alpha_l:0.2,
        b_l:0.5,
        sig_l:0.2,
        t:1.0,
        u_steps:256,
        num_send:batch_size,
        x_min:-max_possible_loss*exposure,
        x_max:0.0
    }
}
fn generate_unif(min:f64, max:f64)->f64{
    min+random::<f64>()*(max-min)
}
fn generate_balance(min_loan_size:f64, max_loan_size:f64)->f64{
    generate_unif(min_loan_size, max_loan_size)
}
fn generate_pd()->f64{
    generate_unif(0.01, 0.05)
}
fn generate_weights(num_macro:usize)->Vec<f64>{
    let rand_weight:Vec<f64>=(0..num_macro).map(|_|random::<f64>()).collect();
    let total:f64=rand_weight.iter().sum();
    rand_weight.into_iter().map(|v|v/total).collect()
}
fn get_loans(
    batch_size:usize, num_macro:usize, 
    min_loan_size:f64, max_loan_size:f64
)->Vec<Loan>{
    (0..batch_size).map(|_index|{
        Loan{
            balance:generate_balance(min_loan_size, max_loan_size),
            lgd:0.5,
            pd:generate_pd(),
            weight:generate_weights(num_macro)
        }
    }).collect()
}
fn main()-> Result<(), io::Error> {
    //let args: Vec<String> = env::args().collect();
    let batch_size:usize=1000;
    let num_batches:usize=1000;
    let min_loan_size=10000.0;
    let max_loan_size=50000.0;
    let x_steps:usize=1024;
    let max_possible_loss=0.14;//more than this an HUGE loss
    let num_macro:usize=3;
    let Parameters{
        lambda,
        q,
        alpha_l,
        b_l,
        sig_l,
        t,
        u_steps,
        num_send:_num_send,
        x_min,
        x_max
    }=get_parameters(
        batch_size, num_batches, 
        min_loan_size, max_loan_size, max_possible_loss
    );
    let liquid_fn=get_liquidity_risk_fn(lambda, q);
    let lgd_fn=get_lgd_cf_fn(alpha_l, b_l, sig_l, t, b_l);//assumption is that it starts at the lgd mean...
    
    let log_loan_cf_fn=log_lpm_cf(
        &lgd_fn, 
        |loan:&Loan|loan.balance*loan.lgd,
        |loan:&Loan|loan.pd,
        |loan:&Loan, index|loan.weight[index]
    );
    let full_exponent=get_full_exponent(
        x_min, x_max, u_steps, num_macro, 
        &liquid_fn, &log_loan_cf_fn
    );
    let generate_loans_and_cf=||{
        let loans=get_loans(batch_size, num_macro, min_loan_size, max_loan_size);
        full_exponent(&loans)
    };
    let mut cf_log=generate_loans_and_cf(); //cf_log has length num_macro*u_steps
    (1..num_batches).for_each(|_|{
        generate_loans_and_cf().iter().enumerate().for_each(|(index, v)|{
            cf_log[index]+=v;
        });
    });
    let y0=vec![0.9, 1.0, 1.1];
    let alpha=vec![0.2, 0.3, 0.2];
    let sigma=vec![0.4, 0.4, 0.3];
    let rho=vec![1.0, -0.4, 0.2, -0.4, 1.0, 0.3, 0.2, 0.3, 1.0];
    let expectation=vasicek::compute_integral_expectation_long_run_one(&y0, &alpha, t);
    let variance=vasicek::compute_integral_variance(&alpha, &sigma, &rho, t);
    let v_mgf=vasicek::get_vasicek_mgf(expectation, variance);

    let final_cf:Vec<Complex<f64>>=cf_log.par_chunks(num_macro).map(v_mgf).collect();//final_cf has length u_steps

    let x_domain:Vec<f64>=fang_oost::get_x_domain(x_steps, x_min, x_max).collect();
    let density:Vec<f64>=fang_oost::get_density(x_min, x_max, fang_oost::get_x_domain(x_steps, x_min, x_max), &final_cf).collect();

    let json_results=json!({"x":x_domain, "density":density});
    let mut file = File::create("docs/loan_density.json")?;
    file.write_all(json_results.to_string().as_bytes())?;

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
