| [Linux][lin-link] | [Codecov][cov-link] |
| :---------------: | :-----------------: |
| ![lin-badge]      | ![cov-badge]        |

[lin-badge]: https://travis-ci.com/phillyfan1138/loan_ec.svg "Travis build status"
[lin-link]:  https://travis-ci.com/phillyfan1138/loan_ec "Travis build status"
[cov-badge]: https://codecov.io/gh/phillyfan1138/loan_ec/branch/master/graph/badge.svg
[cov-link]:  https://codecov.io/gh/phillyfan1138/loan_ec

## Utilities for economic capital assignments for a loan portfolio

This library has a relatively opinionated API for creating a portfolio of loans and performing aggregate statistics (such as loan level risk contributions and expected values).  

## Install

Add the following to your Cargo.toml:

`loan_ec = "0.0.2"`

## Use
A full example is in the [credit_faas_demo](https://github.com/phillyfan1138/credit_faas_demo).

Create instances of the Loan struct:

```rust
extern crate loan_ec;
let loan=Loan{
    balance:1000.0,
    pd:0.03,
    lgd:0.5,
    weight:vec![0.4, 0.6],//must add to one, represents exposure to macro variables
    r:0.5,
    lgd_variance:0.3,
    num:1000.0//number of loans that have these attributes
};
```

Then add to the portfolio:

```rust
let num_u:usize=256;//the higher, the more accurate, but the slower it will run
let x_min=-100000.0;//the minimum of the distribution
let x_max=0.0;//the maximum of the distribution
let mut ec=EconomicCapitalAttributes::new(
    num_u, 
    weight.len()
);
let u_domain:Vec<Complex<f64>>=fang_oost::get_u_domain(
    num_u, x_min, x_max
).collect();

//the characteristic function for the random variable for LGD...in this case, degenerate (a constant)
let lgd_fn=|u:&Complex<f64>, l:f64, _lgd_v:f64|(-u*l).exp();
        
//cf enhancement for ec
let liquid_fn=loan_ec::get_liquidity_risk_fn(lambda, q);

let log_lpm_cf=loan_ec::get_log_lpm_cf(&lgd_fn, &liquid_fn);
ec.process_loan(&loan, &u_domain, &log_lpm_cf);
//keep adding until there are no more loans left...
```

Then to do statistics on the portfolio:

```rust
//variance of macro variables
let variance=vec![0.3, 0.4];
let v_mgf=|u_weights:&[Complex<f64>]|->Complex<f64>{
    u_weights.iter().zip(&variance).map(|(u, v)|{
        -(1.0-v*u).ln()/v
    }).sum::<Complex<f64>>().exp()
};
let final_cf:Vec<Complex<f64>>=ec.get_full_cf(&v_mgf);
```

Using the characteristic function, obtain any number of metrics including expected shortfall and value at risk (from my [cf_dist_utils](https://github.com/phillyfan1138/cf_dist_utils_rust) repository).

```rust
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
```




