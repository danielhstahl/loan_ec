extern crate fang_oost;
extern crate num_complex;
extern crate rayon;
extern crate cf_functions;
extern crate rand;

use rand::prelude::*;
use self::num_complex::Complex;
use self::rayon::prelude::*;

use std::env;
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
    let du=fang_oost::compute_du(x_min, x_max);
    move |loans:&[Loan]|{
        (0..num_u).into_par_iter()
            .flat_map(|u_index|{
                (0..num_systemic).map(|systemic_index|{
                    log_lpm_cf(
                        &get_liquidity(
                            &fang_oost::get_complex_u(fang_oost::get_u(du, u_index))
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
)->impl Fn(&Complex<f64>, &[Loan], usize)-> Complex<f64>//Iterator<Item = Complex<f64> >+std::marker::Sync+std::marker::Sized+std::marker::Send
where
    T:Fn(&Complex<f64>, f64)->Complex<f64> +std::marker::Sync+std::marker::Send,
    U:Fn(&Loan)->f64 +std::marker::Sync+std::marker::Send,
    V:Fn(&Loan)->f64 +std::marker::Sync+std::marker::Send,
    W: Fn(&Loan, usize)->f64+std::marker::Sync+std::marker::Send
{ 
    move |u:&Complex<f64>, loans:&[Loan], index:usize|{
        //(0..num_systemic_factors).map(move |index|{
        loans.iter().fold(Complex::new(0.0, 0.0), |accum, loan|{
            accum+(lgd_cf(u, get_l(loan))-1.0)*get_pd(loan)*get_w(loan, index)
        })
        //})//.collect()//array of size num_systemic_factors
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
    num_weights:usize,
    min_loan_size:f64,
    max_loan_size:f64,
    max_possible_loss:f64
)->Parameters{
    let exposure=get_rough_total_exposure(
        min_loan_size,
        max_loan_size,
        num_batches*batch_size
    );
    Parameters{
        lambda:0.2*exposure,
        q:0.1/exposure,
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
    (0..batch_size).map(|index|{
        Loan{
            balance:generate_balance(min_loan_size, max_loan_size),
            lgd:0.5,
            pd:generate_pd(),
            weight:generate_weights(num_macro)
        }
    }).collect()
}
fn main()-> Result<(), io::Error> {
    let args: Vec<String> = env::args().collect();
    let batch_size:usize=1000;
    let num_batches:usize=1000;
    let num_weights:usize=3;
    let min_loan_size=10000.0;
    let max_loan_size=50000.0;
    let max_possible_loss=0.14;//more than this an HUGE loss
    let num_macro=3;
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
        batch_size, num_batches, num_weights, 
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

    let generate_loans_and_cf=||{
        let loans=get_loans(batch_size, num_macro, min_loan_size, max_loan_size);
        let full_exponent=get_full_exponent(x_min, x_max, u_steps, num_macro, &liquid_fn, &log_loan_cf_fn);
        full_exponent(&loans)
    };
    let mut cf_log=generate_loans_and_cf();
    (1..num_batches).for_each(|_|{
        generate_loans_and_cf().iter().enumerate().for_each(|(index, v)|{
            cf_log[index]+=v;
        });
    });


    
    //println!("num systemic factor {}:", complex_exponent.first().unwrap().len());
    Ok(())
}
