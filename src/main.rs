extern crate fang_oost;
extern crate num_complex;
extern crate rayon;
extern crate cf_functions;
extern crate ws;
use self::num_complex::Complex;
use self::rayon::prelude::*;
use ws::{connect, CloseCode};

fn get_liquidity_risk_fn(
    lambda:f64,
    q:f64
)->impl Fn(&Complex<f64>)->Complex<f64>
{
    move |u:&Complex<f64>|-((-u*lambda).exp()-1.0)*q-u
}

fn get_full_cf_fn(
    x_min:f64,
    x_max:f64,
    get_liquidity:Fn(&Complex<f64>)->Complex<f64>,
    log_lpm_cf:Fn(&Complex<f64>, &[f64])->Complex<f64>
)->impl Fn(&[Complex<f64>], &[f64])->Vec<Complex<f64> > {
    move |cf_discrete:&[Complex<f64>], loans:&[f64]|{
        cf_discrete
            .par_iter()
            .enumerate()
            .map(|(index, cf_increment)|{
                log_lpm_cf(
                    &get_liquidity(
                        //u?
                    ),
                    loans
                ).iter().enumerate().map(|(index, ))
            })
    }
}

fn log_lpm_cf(
    m:usize,
    lgd_cf:Fn(&Complex<f64>, f64)->Complex<f64>,
    get_l:Fn(f64)->f64,
    get_pd:Fn(f64)->f64,
    get_w:Fn(f64, usize)->f64
)->impl Fn(&Complex<f64>, &[f64])->Vec<Complex<f64> >{ 
    move |u:&Complex<f64>, loans:&[f64]|{
        (0..m).map(|index|{ //what is m??
            loans.iter().fold(0.0, |accum, loan|{
                (lgd_cf(u, loan)-1.0)*get_pd(loan)*get_w(loan, index)
            })
        })
    } //should this return a vector or an iterator
}
fn get_lgd_cf_fn(
    lambda:f64,
    theta:f64,
    sigma:f64,
    t:f64,
    x0:f64
)->impl Fn(&Complex<f64>, f64)->Complex<f64>{
   // let exp_t=(-lambda*t).exp();
    //let sig_l=-sigma.powi(2)/(2.0*lambda);
    |u:&Complex<f64>, l:f64|{
        /*let uu=u*l;
        let u_p=uu*(1.0-exp_t)*sig_l+1.0;
        (uu*exp_t*x0/u_p).exp()*u_p.powf(theta/sig_l)*/
        cf_functions::cir_mgf(
            &(u*l), lambda, theta, 
            sigma, t, x0
        )
    }   
}

fn main() {
    if let Err(error) = connect("ws://127.0.0.1:3012", |out| {

        // Queue a message to be sent when the WebSocket is open
        if let Err(_) = out.send("init") {
            println!("Websocket couldn't queue an initial message.")
        } else {
            println!("Client sent message 'Hello WebSocket'. ")
        }

        // The handler needs to take ownership of out, so we use move
        move |msg| {

            // Handle messages received on this connection
            println!("Client got message '{}'. ", msg);

            // Close the connection
            out.close(CloseCode::Normal)
        }

    }) {
        // Inform the user of failure
        println!("Failed to create WebSocket due to: {:?}", error);
    }
    //println!("Hello, world!");
}
