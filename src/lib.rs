//! Economic capital for a loan portfolio.  
//! Based on [my paper](https://github.com/phillyfan1138/CreditRiskExtensions/blob/master/StahlMultiVariatePaper.pdf)
//! on credit economic capital.
//!

use serde_derive::{Deserialize};
use num_complex::Complex;
use rayon::prelude::*;
mod vec_to_mat;


/// Struct representing loan attributes
#[derive(Debug, Deserialize)]
pub struct Loan {
    /// Balance for the loan (in dollars)
    pub balance: f64,
    /// Annualized probability of default.
    /// Can be point in time or through the
    /// cycle.
    pub pd: f64,
    /// Expected value of the loss given
    /// default for the loan.  A positive
    /// number.
    pub lgd: f64,
    /// Vector of weights describing the
    /// exposure of this loan to each
    /// systemic variable.
    pub weight: Vec<f64>,
    #[serde(default = "default_zero")]
    /// Amount of liquidity exposure as a
    /// percent of the balance.  A positive
    /// number.
    pub r: f64,
    #[serde(default = "default_zero")]
    /// Variance of the loss given
    /// default for the loan.
    pub lgd_variance: f64,
    #[serde(default = "default_one")]
    /// The number of loans that have these
    /// features.  Grouping loans into
    /// single Loan structs can vastly
    /// improve computation time.
    pub num: f64,
}

fn default_one() -> f64 {
    1.0
}
fn default_zero() -> f64 {
    0.0
}

/// Returns increment of expected loss for a given loan.
/// A typical use case would have this called multiple
/// times for each loan; once for each weight element
/// in the weight vector of the Loan struct.
///
/// # Arguments
///
/// * `loan` - An instance of the [Loan](struct.Loan.html) struct
/// * `w` - A single element from the weight
/// vector in the Loan struct.
fn get_el_from_loan(loan: &Loan, w: f64) -> f64 {
    -loan.lgd * loan.balance * w * loan.pd * loan.num
}
/// Returns increment of variance for a given loan.
/// A typical use case would have this called multiple
/// times for each loan; once for each weight element
/// in the weight vector of the Loan struct.
///
/// # Arguments
///
/// * `loan` - An instance of the [Loan](struct.Loan.html) struct
/// * `w` - A single element from the weight
/// vector in the Loan struct.
fn get_var_from_loan(loan: &Loan, w: f64) -> f64 {
    (1.0 + loan.lgd_variance) * (loan.lgd * loan.balance).powi(2) * w * loan.pd * loan.num
}
/// Returns incremental "lambda" for a given loan
///
/// # Arguments
///
/// * `loan` - An instance of the [Loan](struct.Loan.html) struct
fn get_lambda_from_loan(loan: &Loan) -> f64 {
    loan.balance * loan.r * loan.num
}
/// Returns risk contribution for a given loan.  This
/// function is used by
/// [experiment_risk_contribution](struct.EconomicCapitalAttributes.html#method.experiment_risk_contribution)
/// but can also be used on its own.  
///
/// # Arguments
///
/// * `loan` - An instance of the [Loan](struct.Loan.html) struct
/// * `el_vec` - The portfolio vector of expected
/// loss.  Has the same size as the weight vector
/// in the Loan struct.
/// * `el_sys` - The vector of expected values for
/// the systemic random variables.  This is
/// typically a vector of ones.
/// * `var_vec` - The portfolio vector of
/// variance.  Has the same size as the weight
/// vector in the Loan struct.
/// * `var_sys` - The vector of variances for the
/// systemic random variables.
/// * `lambda0` - Base loss (in dollars) from a
/// liquidity event.  A positive number.
/// * `lambda` - Sum of r*balance over each loan
/// in the portfolio.
/// * `q` - Probability of liquidity event (scaled
/// by the total portfolio loss).
/// * `c` - Scalar for multiplying the covariance
/// for the risk contribution when using the
/// variance risk measure.  Typically
/// (rho(X)-E\[X\])/sqrt(Var(X)) where rho is the
/// value of the portfolio risk measure (eg, VaR).
pub fn risk_contribution(
    loan: &Loan,
    el_vec: &[f64],
    el_sys: &[f64],
    var_vec: &[f64],
    var_sys: &[f64],
    lambda0: f64,
    lambda: f64,
    q: f64,
    c: f64,
) -> f64 {
    let el_scalar_incremental = 1.0 + q * lambda0;
    let el_scalar_total = q * get_lambda_from_loan(loan);
    let expectation_total = portfolio_expectation(el_vec, el_sys);
    let variance_total = portfolio_variance(el_vec, el_sys, var_vec, var_sys);

    let standard_deviation =
        variance_liquidity(lambda + lambda0, q, expectation_total, variance_total).sqrt();

    let var_scalar_incremental = el_scalar_incremental.powi(2);

    let var_scalar_total = el_scalar_total * (2.0 * el_scalar_incremental + q * lambda);
    let var_el_total = el_scalar_total * (2.0 * lambda0 + lambda);
    let expectation_incremental = el_sys
        .iter()
        .zip(&loan.weight)
        .map(|(e_s, &w)| get_el_from_loan(loan, w) * e_s)
        .sum::<f64>();

    let variance_incremental = el_sys
        .iter()
        .zip(&loan.weight)
        .map(|(el_s, &w)| get_var_from_loan(loan, w) * el_s)
        .sum::<f64>()
        + var_sys
            .iter()
            .zip(&loan.weight)
            .zip(el_vec)
            .map(|((v_s, &w), e_v)| e_v * v_s * get_el_from_loan(loan, w))
            .sum::<f64>();

    el_scalar_incremental * expectation_incremental
        + el_scalar_total * expectation_total
        + c * (var_scalar_incremental * variance_incremental + var_scalar_total * variance_total
            - expectation_incremental * q * lambda0.powi(2)
            - expectation_total * var_el_total)
            / standard_deviation
}
/// Returns the variance of a portfolio with liquidity risk.
///
/// # Arguments
///
/// * `lambda` - Sum of lambda0 (base loss in liquidity event)
/// and the lambda element from the [Loan](struct.Loan.html) struct.
/// * `q` - Probability of liquidity event scaled
/// by the total portfolio loss.
/// * `expectation` - Base expectation for the portfolio
/// without liquidity risk.  Can be computed using the
/// [get_portfolio_expectation](struct.EconomicCapitalAttributes.html#method.get_portfolio_expectation)
/// method.
/// * `variance` - Base variance for the portfolio
/// without liquidity risk.  Can be computed using the
/// [get_portfolio_variance](struct.EconomicCapitalAttributes.html#method.get_portfolio_variance)
/// method.
///
/// # Examples
/// ```
/// extern crate loan_ec;
/// # fn main(){
/// let lambda=1.0;
/// let q=0.01;
/// let expectation=500.0;
/// let variance= 5000.0;
/// let liq_var=loan_ec::variance_liquidity(lambda, q, expectation, variance);
/// # }
/// ```
pub fn variance_liquidity(lambda: f64, q: f64, expectation: f64, variance: f64) -> f64 {
    variance * (1.0 + q * lambda).powi(2) - expectation * q * lambda.powi(2)
}
/// Returns the expectation of a portfolio with liquidity risk
///
/// # Arguments
///
/// * `lambda` - Sum of lambda0 (base loss in liquidity event)
/// and the lambda element from the [Loan](struct.Loan.html) struct.
/// * `q` - Probability of liquidity event (scaled
/// by the total portfolio loss).
/// * `expectation` - Base expectation for the portfolio
/// without liquidity risk.  Can be computed using the
/// [get_portfolio_expectation](struct.EconomicCapitalAttributes.html#method.get_portfolio_expectation)
/// method.
///
/// # Examples
/// ```
/// extern crate loan_ec;
/// # fn main(){
/// let lambda=1.0;
/// let q=0.01;
/// let expectation=500.0;
/// let liq_exp=loan_ec::expectation_liquidity(lambda, q, expectation);
/// # }
/// ```
pub fn expectation_liquidity(lambda: f64, q: f64, expectation: f64) -> f64 {
    expectation * (1.0 + q * lambda)
}

/// Returns a function incorporating liquidity risk to the characteristic
/// function.  This function makes lambda negative: the probability
/// of lambda occurring is -qX since X is negative.
///
/// # Arguments
///
/// * `lambda` - Sum of lambda0 (base loss in liquidity event)
/// and the lambda element from the [Loan](struct.Loan.html) struct.
/// * `q` - Probability of liquidity event (scaled
/// by the total portfolio loss).
pub fn get_liquidity_risk_fn(lambda: f64, q: f64) -> impl Fn(&Complex<f64>) -> Complex<f64> {
    move |u: &Complex<f64>| u - ((-u * lambda).exp() - 1.0) * q
}

/// Returns a function which is the characteristic exponent
/// for a given loan.  The result of this function is used
/// as the third argument in
/// [process_loan](struct.EconomicCapitalAttributes.html#method.process_loan).
///
/// # Arguments
///
/// * `lgd_cf` - The characteristic function for a given
/// loan's loss given default.
/// * `liquidity_cf` - The liquidity function typically
/// instantiated from [get_liquidity_risk_fn](fn.get_liquidity_risk_fn.html).
pub fn get_log_lpm_cf<T, U>(
    lgd_cf: T,
    liquidity_cf: U,
) -> impl Fn(&Complex<f64>, &Loan) -> Complex<f64>
where
    T: Fn(&Complex<f64>, f64, f64) -> Complex<f64>,
    U: Fn(&Complex<f64>) -> Complex<f64>,
{
    move |u: &Complex<f64>, loan: &Loan| {
        (lgd_cf(&liquidity_cf(u), loan.lgd * loan.balance, loan.lgd_variance) - 1.0) * loan.pd
    }
}

/// Holds the attributes for the entire
/// portfolio.
pub struct EconomicCapitalAttributes {
    /// Holds the characteristic function for the portfolio
    /// (without multiplying by the systemic variables).
    /// The only use case for this element is when used as
    /// an input into
    /// [get_experiment_full_cf](struct.EconomicCapitalAttributes.html#method.get_experiment_full_cf).
    pub cf: Vec<Complex<f64>>,
    /// The expected value (first moment) vector of length
    /// num_w for the portfolio.
    pub el_vec: Vec<f64>,
    /// The second moment vector of length num_w for the
    /// portfolio (p_j E[l^2]w_j).
    pub var_vec: Vec<f64>,
    /// The number of systemic random variables.
    pub num_w: usize,
    /// The total liquidity risk for the portfolio.
    /// This is the sum of r*balance over the
    /// portfolio.
    pub lambda: f64,
}
/// Computes portfolio expectation given
/// the incremental vectors of portfolio
/// expectation and systemic expectation.
///
/// # Arguments
///
/// * `el_vec` - The portfolio vector of expected
/// loss.  Has the same size as the weight vector
/// in the [Loan](struct.Loan.html) struct.
/// * `el_sys` - The vector of expected values for
/// the systemic random variables.  This is
/// typically a vector of ones.  Has the same size
/// as the weight vector in the
/// [Loan](struct.Loan.html) struct.
fn portfolio_expectation(el_vec: &[f64], el_sys: &[f64]) -> f64 {
    el_vec
        .iter()
        .zip(el_sys)
        .map(|(el_v, el_s)| el_v * el_s)
        .sum::<f64>()
}
/// Computes portfolio variance given
/// the incremental vectors of portfolio
/// variance and systemic variance.
/// The assumption is that the var_sys
/// are independent.  Otherwise the var_sys
/// needs to be a matrix.
///
/// # Arguments
///
/// * `el_vec` - The portfolio vector of expected
/// loss.  Has the same size as the weight vector
/// in the [Loan](struct.Loan.html) struct.
/// * `el_sys` - The vector of expected values for
/// the systemic random variables.  This is
/// typically a vector of ones.  Has the same size
/// as the weight vector in the
/// [Loan](struct.Loan.html) struct.
/// * `var_vec` - The portfolio vector of
/// variance.  Has the same size as the weight
/// vector in the [Loan](struct.Loan.html) struct.
/// * `var_sys` - The vector of variances for the
/// systemic random variables.  Has the same size
/// as the weight vector in the
/// [Loan](struct.Loan.html) struct.
fn portfolio_variance(el_vec: &[f64], el_sys: &[f64], var_vec: &[f64], var_sys: &[f64]) -> f64 {
    let v_p: f64 = var_vec
        .iter()
        .zip(el_sys)
        .map(|(var_v, el_s)| el_s * var_v)
        .sum::<f64>();
    let e_p: f64 = el_vec
        .iter()
        .zip(var_sys)
        .map(|(el_v, var_s)| el_v.powi(2) * var_s)
        .sum::<f64>();
    v_p + e_p
}
/// Implements economic capital structure
impl EconomicCapitalAttributes {
    /// Creates a new (base) economic capital struct
    /// # Examples
    /// ```
    /// extern crate loan_ec;
    /// # fn main(){
    /// let num_u:usize=100;
    /// let num_w:usize=3;
    /// let ec_attributes=loan_ec::EconomicCapitalAttributes::new(num_u, num_w);
    /// # }
    /// ```
    pub fn new(num_u: usize, num_w: usize) -> Self {
        EconomicCapitalAttributes {
            cf: vec![Complex::new(0.0, 0.0); num_u * num_w],
            el_vec: vec![0.0; num_w],
            var_vec: vec![0.0; num_w],
            num_w,
            lambda: 0.0, // This is sum of r_j*balance_j
        }
    }
    /// Adds a new loan to the portfolio.
    /// Mutates el_vec, var_vec, cf, and lambda.
    ///
    /// # Arguments
    ///
    /// * `loan` - An instance of the [Loan](struct.Loan.html) struct.
    /// * `u_domain` - The vector of complex values
    /// provided to the characteristic function.  Is
    /// typically generated by the
    /// [fang_oost](https://github.com/phillyfan1138/fang_oost_rust)
    /// repository.  See the example.
    /// * `log_lpm_cf` - The result from calling
    /// [get_log_lpm_cf](fn.get_log_lpm_cf.html).
    ///
    /// # Examples
    /// ```
    /// extern crate loan_ec;
    /// extern crate fang_oost;
    /// extern crate rayon;
    /// extern crate num_complex;
    /// use rayon::prelude::*;
    /// use num_complex::Complex;
    /// # fn main(){
    /// let num_u:usize=100;
    /// let num_w:usize=1;
    /// let x_min=-5000.0;
    /// let x_max=0.0;
    /// let mut ec_attributes=loan_ec::EconomicCapitalAttributes::new(num_u, num_w);
    /// let loan=loan_ec::Loan{
    ///     pd: 0.05,
    ///     lgd: 0.5,
    ///     lgd_variance: 0.2,
    ///     balance: 1.0,
    ///     r: 0.0,
    ///     weight: vec![1.0],
    ///     num: 10000.0
    /// };
    /// let u_domain: Vec<Complex<f64>> = fang_oost::get_u_domain(num_u, x_min, x_max).collect();
    /// let lambda=1000.0;
    /// let q=0.0001;
    /// let liquid_fn = loan_ec::get_liquidity_risk_fn(lambda, q);
    /// //the exponent is negative because l represents a loss
    /// let lgd_fn = |u: &Complex<f64>, l: f64, _lgd_v: f64| (-u * l).exp();
    /// let log_lpm_cf = loan_ec::get_log_lpm_cf(&lgd_fn, &liquid_fn);
    /// ec_attributes.process_loan(&loan, &u_domain, &log_lpm_cf);
    /// # }
    /// ```
    pub fn process_loan<U>(&mut self, loan: &Loan, u_domain: &[Complex<f64>], log_lpm_cf: U)
    where
        U: Fn(&Complex<f64>, &Loan) -> Complex<f64> + std::marker::Sync + std::marker::Send,
    {
        let vec_of_cf_u: Vec<Complex<f64>> =
            u_domain.par_iter().map(|u| log_lpm_cf(&u, loan)).collect();
        let num_w = self.num_w;
        self.cf
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, elem)| {
                let row_num = vec_to_mat::get_row_from_index(index, num_w);
                let col_num = vec_to_mat::get_col_from_index(index, num_w);
                *elem += vec_of_cf_u[col_num] * loan.weight[row_num] * loan.num;
            });
        self.el_vec
            .iter_mut()
            .zip(&loan.weight)
            .for_each(|(el, &w)| {
                *el += get_el_from_loan(&loan, w);
            });
        self.var_vec
            .iter_mut()
            .zip(&loan.weight)
            .for_each(|(var, &w)| {
                *var += get_var_from_loan(&loan, w);
            });
        self.lambda += get_lambda_from_loan(&loan);
    }
    /// Performs marginal analytics for a potential loan
    /// to the portfolio.  The typical use case is for
    /// pricing a new loan that could potentially be added
    /// to the portfolio.  For a loan is already in the
    /// portfolio, the
    /// [process_loan](struct.EconomicCapitalAttributes.html#method.process_loan)
    /// method should be used.
    ///
    /// # Arguments
    ///
    /// * `loan` - An instance of the [Loan](struct.Loan.html)
    /// struct.  This is the loan to experiment on.  
    /// * `u_domain` - The vector of complex values
    /// provided to the characteristic function.  Is
    /// typically generated by the
    /// [fang_oost](https://github.com/phillyfan1138/fang_oost_rust)
    /// repository.  See the example.
    /// * `log_lpm_cf` - The result from calling
    /// [get_log_lpm_cf](fn.get_log_lpm_cf.html).
    ///
    /// # Examples
    /// ```
    /// extern crate loan_ec;
    /// extern crate fang_oost;
    /// extern crate rayon;
    /// extern crate num_complex;
    /// use rayon::prelude::*;
    /// use num_complex::Complex;
    /// # fn main(){
    /// let num_u:usize=100;
    /// let num_w:usize=1;
    /// let x_min=-5000.0;
    /// let x_max=0.0;
    /// let q=0.0001;
    /// let lambda=1000.0;
    /// let mut ec_attributes=loan_ec::EconomicCapitalAttributes::new(num_u, num_w);
    /// let loan=loan_ec::Loan{
    ///     pd: 0.05,
    ///     lgd: 0.5,
    ///     lgd_variance: 0.2,
    ///     balance: 1.0,
    ///     r: 0.0,
    ///     weight: vec![1.0],
    ///     num: 10000.0
    /// };
    /// let liquid_fn = loan_ec::get_liquidity_risk_fn(lambda, q);
    /// let u_domain: Vec<Complex<f64>> = fang_oost::get_u_domain(num_u, x_min, x_max).collect();
    /// let lgd_fn = |u: &Complex<f64>, l: f64, _lgd_v: f64| (-u * l).exp();
    /// let log_lpm_cf = loan_ec::get_log_lpm_cf(&lgd_fn, &liquid_fn);
    /// ec_attributes.process_loan(&loan, &u_domain, &log_lpm_cf);
    /// let loan_ec::EconomicCapitalAttributes{
    ///     cf, el_vec, var_vec, lambda, num_w
    /// }=ec_attributes.experiment_loan(
    ///     &loan, &u_domain,
    ///     &log_lpm_cf
    /// );
    /// # }
    /// ```
    pub fn experiment_loan<U>(
        &self,
        loan: &Loan,
        u_domain: &[Complex<f64>],
        log_lpm_cf: U,
    ) -> EconomicCapitalAttributes
    where
        U: Fn(&Complex<f64>, &Loan) -> Complex<f64> + std::marker::Sync + std::marker::Send,
    {
        let vec_of_cf_u: Vec<Complex<f64>> =
            u_domain.par_iter().map(|u| log_lpm_cf(&u, loan)).collect();
        let num_w = self.num_w;
        EconomicCapitalAttributes {
            cf: self
                .cf
                .par_iter()
                .enumerate()
                .map(|(index, elem)| {
                    let row_num = vec_to_mat::get_row_from_index(index, num_w);
                    let col_num = vec_to_mat::get_col_from_index(index, num_w);
                    elem + vec_of_cf_u[col_num] * loan.weight[row_num] * loan.num
                })
                .collect::<Vec<_>>(),
            el_vec: self
                .el_vec
                .iter()
                .zip(&loan.weight)
                .map(|(el, &w)| el + get_el_from_loan(&loan, w))
                .collect::<Vec<_>>(),
            var_vec: self
                .var_vec
                .iter()
                .zip(&loan.weight)
                .map(|(var, &w)| var + get_var_from_loan(&loan, w))
                .collect::<Vec<_>>(),
            lambda: self.lambda + get_lambda_from_loan(&loan),
            num_w,
        }
    }
    /// Finds the risk contribution of a new loan.
    /// This provides a simpler API than obtaining
    /// the results from
    /// [experiment_loan](struct.EconomicCapitalAttributes.html#method.experiment_loan)
    /// and running them through the
    /// [risk_contribution](fn.risk_contribution.html)
    /// function.
    ///
    /// # Arguments
    ///
    /// * `loan` - An instance of the [Loan](struct.Loan.html) struct.  
    /// This is the loan to experiment on.
    /// * `u_domain` - The vector of complex values
    /// provided to the characteristic function.  Is
    /// typically generated by the
    /// [fang_oost](https://github.com/phillyfan1138/fang_oost_rust)
    /// repository.  See the example.
    /// * `log_lpm_cf` - The result from calling
    /// [get_log_lpm_cf](fn.get_log_lpm_cf.html).
    /// * `lambda0` - Base loss (in dollars) from a
    /// liquidity event.  A positive number.
    /// * `q` - Probability of liquidity event (scaled
    /// by the total portfolio loss).
    /// * `mgf_systemic` - Moment generating function
    /// for the systemic random variables.  Typically
    /// a function of el_sys and var_sys.
    /// * `el_sys` - The vector of expected values for
    /// the systemic random variables.  This is
    /// typically a vector of ones.
    /// * `var_sys` - The vector of variances for the
    /// systemic random variables.
    /// * `risk_measure_fn` - Function which computes
    /// the risk measure for the portfolio, e.g. VaR.
    ///
    /// # Examples
    /// ```
    /// extern crate loan_ec;
    /// extern crate fang_oost;
    /// extern crate rayon;
    /// extern crate num_complex;
    /// use rayon::prelude::*;
    /// use num_complex::Complex;
    /// extern crate cf_dist_utils;
    /// # fn main(){
    /// let num_u:usize=100;
    /// let num_w:usize=1;
    /// let x_min=-5000.0;
    /// let x_max=0.0;
    /// let mut ec_attributes=loan_ec::EconomicCapitalAttributes::new(num_u, num_w);
    /// let loan=loan_ec::Loan{
    ///     pd: 0.05,
    ///     lgd: 0.5,
    ///     lgd_variance: 0.2,
    ///     balance: 1.0,
    ///     r: 0.0,
    ///     weight: vec![1.0],
    ///     num: 10000.0
    /// };
    /// let lambda0=1000.0;
    /// let q=0.0001;
    /// let liquid_fn = loan_ec::get_liquidity_risk_fn(lambda0, q);
    /// let u_domain: Vec<Complex<f64>> = fang_oost::get_u_domain(num_u, x_min, x_max).collect();
    /// let lgd_fn = |u: &Complex<f64>, l: f64, _lgd_v: f64| (-u * l).exp();
    /// let log_lpm_cf = loan_ec::get_log_lpm_cf(&lgd_fn, &liquid_fn);
    /// ec_attributes.process_loan(&loan, &u_domain, &log_lpm_cf);
    /// let systemic_expectation = vec![1.0];
    /// let v = vec![0.4];
    /// let quantile=0.01;
    /// let systemic_mgf=|u_weights: &[Complex<f64>]| -> Complex<f64> {
    ///     u_weights
    ///     .iter()
    ///     .zip(&v)
    ///     .map(|(u, v_inst)| -(1.0 - v_inst * u).ln() / v_inst)
    ///     .sum::<Complex<f64>>()
    ///     .exp()
    /// };
    /// let max_iterations = 100;
    /// let tolerance = 0.0001;
    /// let risk_measure_fn = |final_cf: &[Complex<f64>]| {
    ///     let cf_dist_utils::RiskMetric{value_at_risk, ..} =  cf_dist_utils::get_expected_shortfall_and_value_at_risk_discrete_cf(
    ///         quantile,
    ///         x_min,
    ///         x_max,
    ///         max_iterations,
    ///         tolerance,
    ///         final_cf,
    ///     ).unwrap();  
    ///     value_at_risk
    /// };
    /// let rc = ec_attributes.experiment_risk_contribution(
    ///     &loan,
    ///     &u_domain,
    ///     &log_lpm_cf,
    ///     lambda0,
    ///     q,
    ///     &systemic_mgf,
    ///     &systemic_expectation,
    ///     &v,
    ///     &risk_measure_fn,
    /// );
    /// # }
    pub fn experiment_risk_contribution<U, V, T>(
        &self,
        loan: &Loan,
        u_domain: &[Complex<f64>],
        log_lpm_cf: U,
        lambda0: f64,
        q: f64,
        mgf_systemic: V,
        el_sys: &[f64],
        var_sys: &[f64],
        risk_measure_fn: T,
    ) -> f64
    where
        U: Fn(&Complex<f64>, &Loan) -> Complex<f64> + std::marker::Sync + std::marker::Send,
        V: Fn(&[Complex<f64>]) -> Complex<f64> + std::marker::Sync + std::marker::Send,
        T: Fn(&[Complex<f64>]) -> f64 + std::marker::Sync + std::marker::Send,
    {
        let EconomicCapitalAttributes {
            cf,
            el_vec,
            var_vec,
            lambda,
            ..
        } = self.experiment_loan(loan, u_domain, log_lpm_cf);
        let full_cf = self.get_experiment_full_cf(&cf, &mgf_systemic);
        let risk_measure = risk_measure_fn(&full_cf);
        let port_expectation = portfolio_expectation(&el_vec, el_sys);
        let port_variance = portfolio_variance(&el_vec, el_sys, &var_vec, var_sys);
        let liq_expectation = expectation_liquidity(lambda + lambda0, q, port_expectation);
        let liq_variance = variance_liquidity(lambda + lambda0, q, port_expectation, port_variance);
        let c = (risk_measure - liq_expectation) / liq_variance.sqrt();
        risk_contribution(
            loan, &el_vec, el_sys, &var_vec, var_sys, lambda0, lambda, q, c,
        )
    }
    /// Gets the expected value of the portfolio
    /// without liquidity risk.  This should be
    /// called after processing
    /// all the loans in the portfolio.  The
    /// result of this function can be used as
    /// the third argument in
    /// [expectation_liquidity](fn.expectation_liquidity.html)
    /// to obtain the expectation with liquidity
    /// risk.  The result of this function can be
    /// used as the third argument in
    /// [variance_liquidity](fn.variance_liquidity.html)
    /// to obtain the variance with liquidity
    /// risk.
    ///
    /// # Arguments
    ///
    /// * `el_sys` - The vector of expected values for
    /// the systemic random variables.  This is
    /// typically a vector of ones.
    ///
    /// # Examples
    /// ```
    /// extern crate loan_ec;
    /// extern crate fang_oost;
    /// extern crate rayon;
    /// extern crate num_complex;
    /// use rayon::prelude::*;
    /// use num_complex::Complex;
    /// # fn main(){
    /// let num_u:usize=100;
    /// let num_w:usize=1;
    /// let x_min=-5000.0;
    /// let x_max=0.0;
    /// let expectation=vec![1.0];
    /// let lambda=1000.0;
    /// let q=0.0001;
    /// let liquid_fn = loan_ec::get_liquidity_risk_fn(lambda, q);
    /// //the exponent is negative because l represents a loss
    /// let lgd_fn = |u: &Complex<f64>, l: f64, _lgd_v: f64| (-u * l).exp();
    /// let log_lpm_cf = loan_ec::get_log_lpm_cf(&lgd_fn, &liquid_fn);
    /// let mut ec_attributes=loan_ec::EconomicCapitalAttributes::new(num_u, num_w);
    /// let loan=loan_ec::Loan{
    ///     pd: 0.05,
    ///     lgd: 0.5,
    ///     lgd_variance: 0.2,
    ///     balance: 1.0,
    ///     r: 0.0,
    ///     weight: vec![1.0],
    ///     num: 10000.0
    /// };
    /// let u_domain: Vec<Complex<f64>> = fang_oost::get_u_domain(num_u, x_min, x_max).collect();
    /// ec_attributes.process_loan(&loan, &u_domain, &log_lpm_cf);
    /// let expectation_portfolio=ec_attributes.get_portfolio_expectation(&expectation);
    /// # }
    pub fn get_portfolio_expectation(&self, el_sys: &[f64]) -> f64 {
        portfolio_expectation(&self.el_vec, el_sys)
    }
    /// Gets the variance of the portfolio
    /// without liquidity risk.  This should
    /// be called after processing
    /// all the loans in the portfolio.
    /// The result of this function can be
    /// used as the fourth argument in
    /// [variance_liquidity](fn.variance_liquidity.html)
    /// to obtain the variance with liquidity
    /// risk.
    /// # Arguments
    ///
    /// * `el_sys` - The vector of expected values for
    /// the systemic random variables.  This is
    /// typically a vector of ones.
    /// * `var_sys` - The vector of variances for the
    /// systemic random variables.
    ///
    /// # Examples
    /// ```
    /// extern crate loan_ec;
    /// extern crate fang_oost;
    /// extern crate rayon;
    /// extern crate num_complex;
    /// use num_complex::Complex;
    /// use rayon::prelude::*;
    /// # fn main(){
    /// let num_u:usize=100;
    /// let num_w:usize=1;
    /// let x_min=-5000.0;
    /// let x_max=0.0;
    /// let systemic_expectation = vec![1.0];
    /// let v = vec![0.4];
    /// let mut ec_attributes=loan_ec::EconomicCapitalAttributes::new(num_u, num_w);
    /// let loan=loan_ec::Loan{
    ///     pd: 0.05,
    ///     lgd: 0.5,
    ///     lgd_variance: 0.2,
    ///     balance: 1.0,
    ///     r: 0.0,
    ///     weight: vec![1.0],
    ///     num: 10000.0
    /// };
    /// let lambda=1000.0;
    /// let q=0.0001;
    /// let liquid_fn = loan_ec::get_liquidity_risk_fn(lambda, q);
    /// //the exponent is negative because l represents a loss
    /// let lgd_fn = |u: &Complex<f64>, l: f64, _lgd_v: f64| (-u * l).exp();
    /// let log_lpm_cf = loan_ec::get_log_lpm_cf(&lgd_fn, &liquid_fn);
    /// let u_domain: Vec<Complex<f64>> = fang_oost::get_u_domain(num_u, x_min, x_max).collect();
    /// ec_attributes.process_loan(&loan, &u_domain, &log_lpm_cf);
    /// let variance_portfolio=ec_attributes.get_portfolio_variance(&systemic_expectation, &v);
    /// # }
    pub fn get_portfolio_variance(&self, el_sys: &[f64], var_sys: &[f64]) -> f64 {
        portfolio_variance(&self.el_vec, el_sys, &self.var_vec, var_sys)
    }
    /// Merges the loan exponents with the
    /// systemic variables moment generating
    /// function to obtain the discrete
    /// characteristic function for the
    /// portfolio.  This should be called
    /// after obtaining the cf element from
    /// an
    /// [experiment](struct.EconomicCapitalAttributes.html#method.experiment_loan).
    /// If only the risk contribution is
    /// required, consider using the
    /// [experiment_risk_contribution](struct.EconomicCapitalAttributes.html#method.experiment_risk_contribution)
    /// method instead.
    pub fn get_experiment_full_cf<U>(&self, cf: &[Complex<f64>], mgf: &U) -> Vec<Complex<f64>>
    where
        U: Fn(&[Complex<f64>]) -> Complex<f64> + std::marker::Sync + std::marker::Send,
    {
        cf.par_chunks(self.num_w).map(mgf).collect()
    }
    /// Gets the discrete characteristic function
    /// for the portfolio. This should be called
    /// after processing all the loans in the portfolio.
    ///
    /// # Arguments
    ///
    /// * `mgf_systemic` - Moment generating function
    /// for the systemic random variables.  Typically
    /// a function of el_sys and var_sys.
    ///
    /// # Examples
    /// ```
    /// extern crate loan_ec;
    /// extern crate fang_oost;
    /// extern crate rayon;
    /// extern crate num_complex;
    /// use rayon::prelude::*;
    /// use num_complex::Complex;
    /// extern crate cf_dist_utils;
    /// # fn main(){
    /// let num_u:usize=100;
    /// let num_w:usize=1;
    /// let x_min=-5000.0;
    /// let x_max=0.0;
    /// let systemic_expectation = vec![1.0];
    /// let v = vec![0.4];
    /// let lambda=1000.0;
    /// let q=0.0001;
    /// let liquid_fn = loan_ec::get_liquidity_risk_fn(lambda, q);
    /// //the exponent is negative because l represents a loss
    /// let lgd_fn = |u: &Complex<f64>, l: f64, _lgd_v: f64| (-u * l).exp();
    /// let log_lpm_cf = loan_ec::get_log_lpm_cf(&lgd_fn, &liquid_fn);
    /// let mut ec_attributes=loan_ec::EconomicCapitalAttributes::new(num_u, num_w);
    /// let systemic_mgf=|u_weights: &[Complex<f64>]| -> Complex<f64> {
    ///     u_weights
    ///     .iter()
    ///     .zip(&v)
    ///     .map(|(u, v_inst)| -(1.0 - v_inst * u).ln() / v_inst)
    ///     .sum::<Complex<f64>>()
    ///     .exp()
    /// };
    /// let loan=loan_ec::Loan{
    ///     pd: 0.05,
    ///     lgd: 0.5,
    ///     lgd_variance: 0.2,
    ///     balance: 1.0,
    ///     r: 0.0,
    ///     weight: vec![1.0],
    ///     num: 10000.0
    /// };
    /// let u_domain: Vec<Complex<f64>> = fang_oost::get_u_domain(num_u, x_min, x_max).collect();
    /// ec_attributes.process_loan(&loan, &u_domain, &log_lpm_cf);
    /// let cf=ec_attributes.get_full_cf(&systemic_mgf);
    /// # }
    pub fn get_full_cf<U>(&self, mgf_systemic: &U) -> Vec<Complex<f64>>
    where
        U: Fn(&[Complex<f64>]) -> Complex<f64> + std::marker::Sync + std::marker::Send,
    {
        self.get_experiment_full_cf(&self.cf, mgf_systemic)
    }
}

#[cfg(test)]
fn gamma_mgf<'a, 'b: 'a>(variance: &'b [f64]) -> impl Fn(&[Complex<f64>]) -> Complex<f64> + 'a {
    move |u_weights: &[Complex<f64>]| -> Complex<f64> {
        u_weights
            .iter()
            .zip(variance)
            .map(|(u, v)| -(1.0 - v * u).ln() / v)
            .sum::<Complex<f64>>()
            .exp()
    }
}

#[cfg(test)]
fn test_mgf(u_weights: &[Complex<f64>]) -> Complex<f64> {
    u_weights.iter().sum::<Complex<f64>>().exp()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::*;
    #[test]
    fn construct_hold_discrete_cf() {
        let discrete_cf = EconomicCapitalAttributes::new(256, 3);
        let cf = discrete_cf.cf;
        assert_eq!(cf.len(), 256 * 3);
        assert_eq!(cf[0], Complex::new(0.0, 0.0)); //first three should be the same "u"
        assert_eq!(cf[1], Complex::new(0.0, 0.0));
        assert_eq!(cf[2], Complex::new(0.0, 0.0));
    }
    #[test]
    fn test_process_loan() {
        let mut discrete_cf = EconomicCapitalAttributes::new(256, 3);
        let loan = Loan {
            pd: 0.05,
            lgd: 0.5,
            r: 0.0,
            balance: 1000.0,
            lgd_variance: 0.0,
            weight: vec![0.5, 0.5, 0.5],
            num: 1.0,
        };
        let log_lpm_cf = |_u: &Complex<f64>, _loan: &Loan| Complex::new(1.0, 0.0);
        let u_domain: Vec<Complex<f64>> = fang_oost::get_u_domain(256, 0.0, 1.0).collect();
        discrete_cf.process_loan(&loan, &u_domain, &log_lpm_cf);
        let cf = discrete_cf.cf;
        assert_eq!(cf.len(), 256 * 3);
        cf.iter().for_each(|cf_el| {
            assert_eq!(cf_el, &Complex::new(0.5 as f64, 0.0 as f64));
        });
    }
    #[test]
    fn test_process_loans_with_final() {
        let mut discrete_cf = EconomicCapitalAttributes::new(256, 3);
        let loan = Loan {
            pd: 0.05,
            lgd: 0.5,
            balance: 1000.0,
            lgd_variance: 0.0,
            r: 0.0,
            weight: vec![0.5, 0.5, 0.5],
            num: 1.0,
        };
        let u_domain: Vec<Complex<f64>> = fang_oost::get_u_domain(256, 0.0, 1.0).collect();
        let log_lpm_cf = |_u: &Complex<f64>, _loan: &Loan| Complex::new(1.0, 0.0);
        discrete_cf.process_loan(&loan, &u_domain, &log_lpm_cf);
        let final_cf: Vec<Complex<f64>> = discrete_cf.get_full_cf(&test_mgf);

        assert_eq!(final_cf.len(), 256);
        final_cf.iter().for_each(|cf_el| {
            assert_eq!(cf_el, &Complex::new(1.5 as f64, 0.0 as f64).exp());
        });
    }
    #[test]
    fn test_actually_get_density() {
        let x_min = -6000.0;
        let x_max = 0.0;
        let mut discrete_cf = EconomicCapitalAttributes::new(256, 1);
        let lambda = 1000.0;
        let q = 0.0001;
        let liquid_fn = get_liquidity_risk_fn(lambda, q);

        let u_domain: Vec<Complex<f64>> = fang_oost::get_u_domain(256, x_min, x_max).collect();
        let lgd_fn = |u: &Complex<f64>, l: f64, _lgd_v: f64| (-u * l).exp();
        let log_lpm_cf = get_log_lpm_cf(&lgd_fn, &liquid_fn);

        let loan = Loan {
            pd: 0.05,
            lgd: 0.5,
            lgd_variance: 0.0, //doesnt matter for this test
            balance: 1.0,
            r: 0.0,
            weight: vec![1.0],
            num: 10000.0,
        };
        discrete_cf.process_loan(&loan, &u_domain, &log_lpm_cf);
        let v = vec![0.3];
        let v_mgf = gamma_mgf(&v);
        let final_cf: Vec<Complex<f64>> = discrete_cf.get_full_cf(&v_mgf);

        assert_eq!(final_cf.len(), 256);
        let max_iterations = 100;
        let tolerance = 0.0001;
        let cf_dist_utils::RiskMetric{expected_shortfall, value_at_risk} = cf_dist_utils::get_expected_shortfall_and_value_at_risk_discrete_cf(
            0.01,
            x_min,
            x_max,
            max_iterations,
            tolerance,
            &final_cf,
        ).unwrap();
        assert!(expected_shortfall > value_at_risk);
    }
    #[test]
    fn test_compare_expected_value() {
        let balance = 1.0;
        let pd = 0.05;
        let lgd = 0.5;
        let num_loans = 10000.0;
        let lambda = 1000.0; //loss in the event of a liquidity crisis
        let q = 0.01 / (num_loans * pd * lgd * balance);
        let expectation = -pd * lgd * balance * (1.0 + lambda * q) * num_loans;
        let x_min = (expectation - lambda) * 3.0;
        let x_max = 0.0;
        let num_u: usize = 1024;
        let mut discrete_cf = EconomicCapitalAttributes::new(num_u, 1);

        let liquid_fn = get_liquidity_risk_fn(lambda, q);

        //the exponent is negative because l represents a loss
        let lgd_fn = |u: &Complex<f64>, l: f64, _lgd_v: f64| (-u * l).exp();

        let u_domain: Vec<Complex<f64>> = fang_oost::get_u_domain(num_u, x_min, x_max).collect();
        let log_lpm_cf = get_log_lpm_cf(&lgd_fn, &liquid_fn);

        let loan = Loan {
            pd,
            lgd,
            balance,
            r: 0.0,
            lgd_variance: 0.0,
            weight: vec![1.0],
            num: num_loans, //homogenous
        };
        discrete_cf.process_loan(&loan, &u_domain, &log_lpm_cf);
        let v = vec![0.3];
        let v_mgf = gamma_mgf(&v);
        let final_cf: Vec<Complex<f64>> = discrete_cf.get_full_cf(&v_mgf);
        assert_eq!(final_cf.len(), num_u);
        let expectation_approx =
            cf_dist_utils::get_expectation_discrete_cf(x_min, x_max, &final_cf);

        assert_abs_diff_eq!(expectation_approx, expectation, epsilon = 0.00001);
        assert_abs_diff_eq!(
            expectation_liquidity(lambda, q, discrete_cf.get_portfolio_expectation(&vec![1.0])),
            expectation,
            epsilon = 0.00001
        );
    }
    #[test]
    fn test_compare_expected_value_and_variance_no_stochastic_lgd() {
        let balance = 1.0;
        let pd = 0.05;
        let lgd = 0.5;
        let num_loans = 10000.0;
        let lambda = 1000.0; //loss in the event of a liquidity crisis
        let q = 0.01 / (num_loans * pd * lgd * balance);
        let x_min = (-num_loans * pd * lgd * balance - lambda) * 3.0;

        let v = vec![0.4, 0.3];
        //let v2=vec![0.4, 0.3];
        let systemic_expectation = vec![1.0, 1.0];
        let v_mgf = gamma_mgf(&v);

        let weight = vec![0.4, 0.6];

        let x_max = 0.0;
        let num_u: usize = 1024;
        let mut discrete_cf = EconomicCapitalAttributes::new(num_u, v.len());

        let liquid_fn = get_liquidity_risk_fn(lambda, q);

        //the exponent is negative because l represents a loss
        let lgd_fn = |u: &Complex<f64>, l: f64, _lgd_v: f64| (-u * l).exp();
        let u_domain: Vec<Complex<f64>> = fang_oost::get_u_domain(num_u, x_min, x_max).collect();
        let log_lpm_cf = get_log_lpm_cf(&lgd_fn, &liquid_fn);

        let loan = Loan {
            pd,
            lgd,
            r: 0.0,
            balance,
            weight,
            lgd_variance: 0.0,
            num: num_loans, //homogenous
        };
        discrete_cf.process_loan(&loan, &u_domain, &log_lpm_cf);

        let expectation = discrete_cf.get_portfolio_expectation(&systemic_expectation);
        let variance = discrete_cf.get_portfolio_variance(&systemic_expectation, &v);
        let expectation_liquid = expectation_liquidity(lambda, q, expectation);
        let variance_liquid = variance_liquidity(lambda, q, expectation, variance);

        let final_cf: Vec<Complex<f64>> = discrete_cf.get_full_cf(&v_mgf);
        assert_eq!(final_cf.len(), num_u);
        let expectation_approx =
            cf_dist_utils::get_expectation_discrete_cf(x_min, x_max, &final_cf);
        let variance_approx = cf_dist_utils::get_variance_discrete_cf(x_min, x_max, &final_cf);

        assert_abs_diff_eq!(expectation_approx, expectation_liquid, epsilon = 0.00001);
        assert_abs_diff_eq!(variance_approx, variance_liquid, epsilon = 0.1);
    }
    #[test]
    fn test_compare_expected_value_and_variance_stochastic_lgd() {
        let balance = 1.0;
        let pd = 0.05;
        let lgd = 0.5;
        let num_loans = 10000.0;
        let lambda = 1000.0; //loss in the event of a liquidity crisis
        let q = 0.01 / (num_loans * pd * lgd * balance);
        let x_min = (-num_loans * pd * lgd * balance - lambda) * 3.0;
        let v = vec![0.4, 0.3];
        let systemic_expectation = vec![1.0, 1.0];
        let v_mgf = gamma_mgf(&v);
        let lgd_variance = 0.2;
        let weight = vec![0.4, 0.6];

        let x_max = 0.0;
        let num_u: usize = 1024;
        let mut discrete_cf = EconomicCapitalAttributes::new(num_u, v.len());

        let liquid_fn = get_liquidity_risk_fn(lambda, q);

        //the exponent is negative because l represents a loss
        let lgd_fn = |u: &Complex<f64>, l: f64, lgd_v: f64| {
            cf_functions::gamma_cf(&(-u * l), 1.0 / lgd_v, lgd_v)
        };
        let u_domain: Vec<Complex<f64>> = fang_oost::get_u_domain(num_u, x_min, x_max).collect();
        let log_lpm_cf = get_log_lpm_cf(&lgd_fn, &liquid_fn);

        let loan = Loan {
            pd,
            lgd,
            balance,
            r: 0.0,
            lgd_variance,
            weight,
            num: num_loans, //homogenous
        };
        discrete_cf.process_loan(&loan, &u_domain, &log_lpm_cf);

        let expectation = discrete_cf.get_portfolio_expectation(&systemic_expectation);
        let variance = discrete_cf.get_portfolio_variance(&systemic_expectation, &v);

        let expectation_liquid = expectation_liquidity(lambda, q, expectation);
        let variance_liquid = variance_liquidity(lambda, q, expectation, variance);
        let final_cf: Vec<Complex<f64>> = discrete_cf.get_full_cf(&v_mgf);
        assert_eq!(final_cf.len(), num_u);
        let expectation_approx =
            cf_dist_utils::get_expectation_discrete_cf(x_min, x_max, &final_cf);
        let variance_approx = cf_dist_utils::get_variance_discrete_cf(x_min, x_max, &final_cf);

        assert_abs_diff_eq!(expectation_approx, expectation_liquid, epsilon = 0.00001);
        assert_abs_diff_eq!(variance_approx, variance_liquid, epsilon = 0.1);
    }
    #[test]
    fn test_compare_expected_value_and_variance_stochastic_lgd_non_homogenous() {
        let balance1 = 1.0;
        let balance2 = 1.5;
        let pd1 = 0.05;
        let pd2 = 0.03;
        let lgd1 = 0.5;
        let lgd2 = 0.6;
        let num_loans = 5000.0;
        let lambda = 1000.0; //loss in the event of a liquidity crisis
        let q = 0.01 / (num_loans * pd1 * lgd1 * balance1 * 2.0);
        let x_min = (-num_loans * pd1 * lgd1 * balance1 * 2.0 - lambda) * 3.0;
        let v = vec![0.4, 0.3];
        //let v2=vec![0.4, 0.3];
        let systemic_expectation = vec![1.0, 1.0];
        let v_mgf = gamma_mgf(&v);
        let lgd_variance = 0.2;
        let weight1 = vec![0.4, 0.6];
        let weight2 = vec![0.3, 0.7];

        let x_max = 0.0;
        let num_u: usize = 1024;
        let mut discrete_cf = EconomicCapitalAttributes::new(num_u, v.len());

        let liquid_fn = get_liquidity_risk_fn(lambda, q);

        //the exponent is negative because l represents a loss
        let lgd_fn = |u: &Complex<f64>, l: f64, lgd_v: f64| {
            cf_functions::gamma_cf(&(-u * l), 1.0 / lgd_v, lgd_v)
        };
        let u_domain: Vec<Complex<f64>> = fang_oost::get_u_domain(num_u, x_min, x_max).collect();
        let log_lpm_cf = get_log_lpm_cf(&lgd_fn, &liquid_fn);

        let loan1 = Loan {
            pd: pd1,
            lgd: lgd1,
            balance: balance1,
            r: 0.0,
            lgd_variance,
            weight: weight1,
            num: num_loans, //homogenous
        };
        let loan2 = Loan {
            pd: pd2,
            lgd: lgd2,
            balance: balance2,
            r: 0.0,
            lgd_variance,
            weight: weight2,
            num: num_loans, //homogenous
        };
        discrete_cf.process_loan(&loan1, &u_domain, &log_lpm_cf);
        discrete_cf.process_loan(&loan2, &u_domain, &log_lpm_cf);

        let expectation = discrete_cf.get_portfolio_expectation(&systemic_expectation);
        let variance = discrete_cf.get_portfolio_variance(&systemic_expectation, &v);

        let expectation_liquid = expectation_liquidity(lambda, q, expectation);
        let variance_liquid = variance_liquidity(lambda, q, expectation, variance);
        let final_cf: Vec<Complex<f64>> = discrete_cf.get_full_cf(&v_mgf);
        assert_eq!(final_cf.len(), num_u);
        let expectation_approx =
            cf_dist_utils::get_expectation_discrete_cf(x_min, x_max, &final_cf);
        let variance_approx = cf_dist_utils::get_variance_discrete_cf(x_min, x_max, &final_cf);

        assert_abs_diff_eq!(expectation_approx, expectation_liquid, epsilon = 0.00001);
        assert_abs_diff_eq!(variance_approx, variance_liquid, epsilon = 0.1);
    }
    #[test]
    fn test_compare_expected_value_and_variance_stochastic_lgd_non_homogenous_large_w() {
        let balance1 = 1.0;
        let balance2 = 1.5;
        let pd1 = 0.05;
        let pd2 = 0.03;
        let lgd1 = 0.5;
        let lgd2 = 0.6;
        let num_loans = 5000.0;
        let lambda = 1000.0; //loss in the event of a liquidity crisis
        let q = 0.01 / (num_loans * pd1 * lgd1 * balance1 * 2.0);
        let x_min = (-num_loans * pd1 * lgd1 * balance1 * 2.0 - lambda) * 5.0;
        let v = vec![0.4, 0.3, 4.0, 3.0];
        let systemic_expectation = vec![1.0, 1.0, 1.0, 1.0];
        let v_mgf = gamma_mgf(&v);
        let lgd_variance = 0.2;
        let weight1 = vec![0.2, 0.3, 0.3, 0.2];
        let weight2 = vec![0.3, 0.2, 0.2, 0.3];

        let x_max = 0.0;
        let num_u: usize = 1024;
        let mut discrete_cf = EconomicCapitalAttributes::new(num_u, v.len());

        let liquid_fn = get_liquidity_risk_fn(lambda, q);

        //the exponent is negative because l represents a loss
        let lgd_fn = |u: &Complex<f64>, l: f64, lgd_v: f64| {
            cf_functions::gamma_cf(&(-u * l), 1.0 / lgd_v, lgd_v)
        };
        let u_domain: Vec<Complex<f64>> = fang_oost::get_u_domain(num_u, x_min, x_max).collect();
        let log_lpm_cf = get_log_lpm_cf(&lgd_fn, &liquid_fn);

        let loan1 = Loan {
            pd: pd1,
            lgd: lgd1,
            balance: balance1,
            r: 0.0,
            lgd_variance,
            weight: weight1,
            num: num_loans, //homogenous
        };
        let loan2 = Loan {
            pd: pd2,
            lgd: lgd2,
            balance: balance2,
            r: 0.0,
            lgd_variance,
            weight: weight2,
            num: num_loans, //homogenous
        };
        discrete_cf.process_loan(&loan1, &u_domain, &log_lpm_cf);
        discrete_cf.process_loan(&loan2, &u_domain, &log_lpm_cf);

        let expectation = discrete_cf.get_portfolio_expectation(&systemic_expectation);
        let variance = discrete_cf.get_portfolio_variance(&systemic_expectation, &v);

        let expectation_liquid = expectation_liquidity(lambda, q, expectation);
        let variance_liquid = variance_liquidity(lambda, q, expectation, variance);
        let final_cf: Vec<Complex<f64>> = discrete_cf.get_full_cf(&v_mgf);
        assert_eq!(final_cf.len(), num_u);
        let expectation_approx =
            cf_dist_utils::get_expectation_discrete_cf(x_min, x_max, &final_cf);
        let variance_approx = cf_dist_utils::get_variance_discrete_cf(x_min, x_max, &final_cf);

        assert_abs_diff_eq!(expectation_approx, expectation_liquid, epsilon = 0.00001);
        assert_abs_diff_eq!(variance_approx, variance_liquid, epsilon = 0.1);
    }
    #[test]
    fn test_basic_risk_contribution() {
        let balance = 1.0;
        let pd = 0.05;
        let lgd = 0.5;
        let num_loans = 9999.0;
        let lambda = 0.0; //loss in the event of a liquidity crisis
        let q = 0.0 / (num_loans * pd * lgd * balance);
        let x_min = (-num_loans * pd * lgd * balance - lambda) * 3.0;
        let v = vec![0.4, 0.3];
        //let v2=vec![0.4, 0.3];
        let systemic_expectation = vec![1.0, 1.0];
        let v_mgf = gamma_mgf(&v);
        let lgd_variance = 0.2;
        let weight1 = vec![0.4, 0.6];
        let weight2 = vec![0.4, 0.6];

        let x_max = 0.0;
        let num_u: usize = 1024;
        let mut discrete_cf = EconomicCapitalAttributes::new(num_u, v.len());

        let liquid_fn = get_liquidity_risk_fn(lambda, q);

        //the exponent is negative because l represents a loss
        let lgd_fn = |u: &Complex<f64>, l: f64, lgd_v: f64| {
            cf_functions::gamma_cf(&(-u * l), 1.0 / lgd_v, lgd_v)
        };
        let u_domain: Vec<Complex<f64>> = fang_oost::get_u_domain(num_u, x_min, x_max).collect();
        let log_lpm_cf = get_log_lpm_cf(&lgd_fn, &liquid_fn);

        let loan = Loan {
            pd,
            lgd,
            balance,
            lgd_variance,
            weight: weight1,
            r: 0.0,
            num: num_loans, //homogenous
        };
        discrete_cf.process_loan(&loan, &u_domain, &log_lpm_cf);

        let final_cf: Vec<Complex<f64>> = discrete_cf.get_full_cf(&v_mgf);
        assert_eq!(final_cf.len(), num_u);

        let new_loan = Loan {
            pd,
            lgd,
            balance,
            lgd_variance,
            weight: weight2,
            r: 0.0,
            num: 1.0,
        };

        let c = 5.0; //arbitrary
        let EconomicCapitalAttributes {
            el_vec, var_vec, ..
        } = discrete_cf.experiment_loan(&new_loan, &u_domain, &log_lpm_cf);
        let new_variance = portfolio_variance(&el_vec, &systemic_expectation, &var_vec, &v);
        let new_expectation = portfolio_expectation(&el_vec, &systemic_expectation);
        let rc = risk_contribution(
            &new_loan,
            &el_vec,
            &systemic_expectation,
            &var_vec,
            &v,
            0.0,
            0.0,
            q,
            c,
        );
        assert_abs_diff_eq!(
            rc * 10000.0,
            new_expectation + c * new_variance.sqrt(),
            epsilon = 0.1
        );
    }
    #[test]
    fn test_lambda_risk_contribution() {
        let balance = 1.0;
        let pd = 0.05;
        let lgd = 0.5;
        let num_loans = 9999.0;
        let r = 0.0;
        let lambda0 = 1000.0; //loss in the event of a liquidity crisis
        let q = 0.01 / (num_loans * pd * lgd * balance);
        let x_min = (-num_loans * pd * lgd * balance - lambda0) * 3.0;
        let v = vec![0.4, 0.3];
        //let v2=vec![0.4, 0.3];
        let systemic_expectation = vec![1.0, 1.0];
        let v_mgf = gamma_mgf(&v);
        let lgd_variance = 0.2;
        let weight1 = vec![0.4, 0.6];
        let weight2 = vec![0.4, 0.6];

        let x_max = 0.0;
        let num_u: usize = 1024;
        let mut discrete_cf = EconomicCapitalAttributes::new(num_u, v.len());

        let liquid_fn = get_liquidity_risk_fn(lambda0, q);

        //the exponent is negative because l represents a loss
        let lgd_fn = |u: &Complex<f64>, l: f64, lgd_v: f64| {
            cf_functions::gamma_cf(&(-u * l), 1.0 / lgd_v, lgd_v)
        };
        let u_domain: Vec<Complex<f64>> = fang_oost::get_u_domain(num_u, x_min, x_max).collect();
        let log_lpm_cf = get_log_lpm_cf(&lgd_fn, &liquid_fn);

        let loan = Loan {
            pd,
            lgd,
            balance,
            lgd_variance,
            weight: weight1,
            r,
            num: num_loans, //homogenous
        };
        discrete_cf.process_loan(&loan, &u_domain, &log_lpm_cf);

        let final_cf: Vec<Complex<f64>> = discrete_cf.get_full_cf(&v_mgf);
        assert_eq!(final_cf.len(), num_u);

        let new_loan = Loan {
            pd,
            lgd,
            balance,
            lgd_variance,
            weight: weight2,
            r,
            num: 1.0,
        };

        let c = 5.0; //arbitrary
        let EconomicCapitalAttributes {
            el_vec,
            var_vec,
            lambda,
            ..
        } = discrete_cf.experiment_loan(&new_loan, &u_domain, &log_lpm_cf);
        let new_variance = portfolio_variance(&el_vec, &systemic_expectation, &var_vec, &v);
        let new_expectation = portfolio_expectation(&el_vec, &systemic_expectation);
        let liquid_exp = expectation_liquidity(lambda0, q, new_expectation);
        let liquid_var = variance_liquidity(lambda0, q, new_expectation, new_variance);
        assert_abs_diff_eq!(lambda, 0.0, epsilon = 0.00000001);
        let rc = risk_contribution(
            &new_loan,
            &el_vec,
            &systemic_expectation,
            &var_vec,
            &v,
            lambda0,
            0.0,
            q,
            c,
        );
        assert_abs_diff_eq!(
            rc * (num_loans + 1.0),
            liquid_exp + c * liquid_var.sqrt(),
            epsilon = 0.1
        );
    }
    #[test]
    fn test_lambda_qzero_risk_contribution() {
        let balance = 1.0;
        let pd = 0.05;
        let lgd = 0.5;
        let num_loans = 9999.0;
        let r = 0.2;
        let lambda0 = 1000.0; //loss in the event of a liquidity crisis
        let q = 0.0 / (num_loans * pd * lgd * balance);
        let x_min = (-num_loans * pd * lgd * balance - lambda0) * 3.0;
        let v = vec![0.4, 0.3];
        //let v2=vec![0.4, 0.3];
        let systemic_expectation = vec![1.0, 1.0];
        let v_mgf = gamma_mgf(&v);
        let lgd_variance = 0.2;
        let weight1 = vec![0.4, 0.6];
        let weight2 = vec![0.4, 0.6];

        let x_max = 0.0;
        let num_u: usize = 1024;
        let mut discrete_cf = EconomicCapitalAttributes::new(num_u, v.len());

        let liquid_fn = get_liquidity_risk_fn(lambda0, q);

        //the exponent is negative because l represents a loss
        let lgd_fn = |u: &Complex<f64>, l: f64, lgd_v: f64| {
            cf_functions::gamma_cf(&(-u * l), 1.0 / lgd_v, lgd_v)
        };
        let u_domain: Vec<Complex<f64>> = fang_oost::get_u_domain(num_u, x_min, x_max).collect();
        let log_lpm_cf = get_log_lpm_cf(&lgd_fn, &liquid_fn);

        let loan = Loan {
            pd,
            lgd,
            balance,
            lgd_variance,
            weight: weight1,
            r,
            num: num_loans, //homogenous
        };
        discrete_cf.process_loan(&loan, &u_domain, &log_lpm_cf);

        let final_cf: Vec<Complex<f64>> = discrete_cf.get_full_cf(&v_mgf);
        assert_eq!(final_cf.len(), num_u);

        let new_loan = Loan {
            pd,
            lgd,
            balance,
            lgd_variance,
            weight: weight2,
            r,
            num: 1.0,
        };

        let c = 5.0; //arbitrary
        let EconomicCapitalAttributes {
            el_vec, var_vec, ..
        } = discrete_cf.experiment_loan(&new_loan, &u_domain, &log_lpm_cf);
        let new_variance = portfolio_variance(&el_vec, &systemic_expectation, &var_vec, &v);
        let new_expectation = portfolio_expectation(&el_vec, &systemic_expectation);
        let liquid_exp = expectation_liquidity(lambda0, q, new_expectation);
        let liquid_var = variance_liquidity(lambda0, q, new_expectation, new_variance);
        let rc = risk_contribution(
            &new_loan,
            &el_vec,
            &systemic_expectation,
            &var_vec,
            &v,
            lambda0,
            0.0,
            q,
            c,
        );
        assert_abs_diff_eq!(
            rc * (num_loans + 1.0),
            liquid_exp + c * liquid_var.sqrt(),
            epsilon = 0.1
        );
    }
    #[test]
    fn test_lambda_incremental_risk_contribution() {
        let balance = 1.0;
        let pd = 0.05;
        let lgd = 0.5;
        let num_loans = 9999.0;
        let r = 0.1;
        let lambda0 = 100.0; //loss in the event of a liquidity crisis
        let lambda = r * balance * (num_loans + 1.0);
        let q = 0.01 / (num_loans * pd * lgd * balance);
        let x_min = (-num_loans * pd * lgd * balance - lambda0 - lambda) * 3.0;
        let v = vec![0.4, 0.3];
        //let v2=vec![0.4, 0.3];
        let systemic_expectation = vec![1.0, 1.0];
        let v_mgf = gamma_mgf(&v);
        let lgd_variance = 0.2;
        let weight1 = vec![0.4, 0.6];
        let weight2 = vec![0.4, 0.6];

        let x_max = 0.0;
        let num_u: usize = 1024;
        let mut discrete_cf = EconomicCapitalAttributes::new(num_u, v.len());

        let liquid_fn = get_liquidity_risk_fn(lambda0 + lambda, q);

        //the exponent is negative because l represents a loss
        let lgd_fn = |u: &Complex<f64>, l: f64, lgd_v: f64| {
            cf_functions::gamma_cf(&(-u * l), 1.0 / lgd_v, lgd_v)
        };
        let u_domain: Vec<Complex<f64>> = fang_oost::get_u_domain(num_u, x_min, x_max).collect();
        let log_lpm_cf = get_log_lpm_cf(&lgd_fn, &liquid_fn);

        let loan = Loan {
            pd,
            lgd,
            balance,
            lgd_variance,
            weight: weight1,
            r,
            num: num_loans, //homogenous
        };
        discrete_cf.process_loan(&loan, &u_domain, &log_lpm_cf);

        let final_cf: Vec<Complex<f64>> = discrete_cf.get_full_cf(&v_mgf);
        assert_eq!(final_cf.len(), num_u);

        let new_loan = Loan {
            pd,
            lgd,
            balance,
            lgd_variance,
            weight: weight2,
            r,
            num: 1.0,
        };

        let c = 5.0; //arbitrary
        let EconomicCapitalAttributes {
            el_vec,
            var_vec,
            lambda: lambda_new,
            ..
        } = discrete_cf.experiment_loan(&new_loan, &u_domain, &log_lpm_cf);
        let new_variance = portfolio_variance(&el_vec, &systemic_expectation, &var_vec, &v);
        let new_expectation = portfolio_expectation(&el_vec, &systemic_expectation);
        let liquid_exp = expectation_liquidity(lambda + lambda0, q, new_expectation);
        let liquid_var = variance_liquidity(lambda + lambda0, q, new_expectation, new_variance);

        assert_abs_diff_eq!(lambda, lambda_new, epsilon = 0.0000001);

        let rc = risk_contribution(
            &new_loan,
            &el_vec,
            &systemic_expectation,
            &var_vec,
            &v,
            lambda0,
            lambda,
            q,
            c,
        );
        assert_abs_diff_eq!(
            rc * (num_loans + 1.0),
            liquid_exp + c * liquid_var.sqrt(),
            epsilon = 0.1
        );
    }
    #[test]
    fn test_lambda_incremental_risk_contribution_non_homogenous() {
        let balance1 = 1.0;
        let balance2 = 2.0;
        let pd1 = 0.05;
        let pd2 = 0.03;
        let lgd1 = 0.5;
        let lgd2 = 0.4;
        let num_loans1 = 6000.0;
        let num_loans2 = 4000.0;
        let r1 = 0.1;
        let r2 = 0.2;
        let lambda0 = 100.0; //loss in the event of a liquidity crisis
        let lambda = r1 * balance1 * num_loans1 + r2 * balance2 * num_loans2;
        let q = 0.01 / (num_loans1 * pd1 * lgd1 * balance1 + num_loans2 * pd2 * lgd2 * balance2);
        let x_min = (-num_loans1 * pd1 * lgd1 * balance1
            - num_loans2 * pd2 * lgd2 * balance2
            - lambda0
            - lambda)
            * 3.0;
        let v = vec![0.4, 0.3];
        let systemic_expectation = vec![1.0, 1.0];
        let lgd_variance = 0.2;
        let weight1_1 = vec![0.4, 0.6];
        let weight2_1 = vec![0.3, 0.7];

        let x_max = 0.0;
        let num_u: usize = 1024;
        let mut discrete_cf = EconomicCapitalAttributes::new(num_u, v.len());

        let liquid_fn = get_liquidity_risk_fn(lambda0 + lambda, q);

        //the exponent is negative because l represents a loss
        let lgd_fn = |u: &Complex<f64>, l: f64, lgd_v: f64| {
            cf_functions::gamma_cf(&(-u * l), 1.0 / lgd_v, lgd_v)
        };
        let u_domain: Vec<Complex<f64>> = fang_oost::get_u_domain(num_u, x_min, x_max).collect();
        let log_lpm_cf = get_log_lpm_cf(&lgd_fn, &liquid_fn);

        let loan1 = Loan {
            pd: pd1,
            lgd: lgd1,
            balance: balance1,
            lgd_variance,
            weight: weight1_1,
            r: r1,
            num: num_loans1, //homogenous
        };
        discrete_cf.process_loan(&loan1, &u_domain, &log_lpm_cf);

        let loan2 = Loan {
            pd: pd2,
            lgd: lgd2,
            balance: balance2,
            lgd_variance,
            weight: weight2_1,
            r: r2,
            num: num_loans2,
        };

        let c = 5.0; //arbitrary
        let EconomicCapitalAttributes {
            el_vec,
            var_vec,
            lambda: lambda_new,
            ..
        } = discrete_cf.experiment_loan(&loan2, &u_domain, &log_lpm_cf);
        let new_variance = portfolio_variance(&el_vec, &systemic_expectation, &var_vec, &v);
        let new_expectation = portfolio_expectation(&el_vec, &systemic_expectation);
        let liquid_exp = expectation_liquidity(lambda + lambda0, q, new_expectation);
        let liquid_var = variance_liquidity(lambda + lambda0, q, new_expectation, new_variance);

        assert_abs_diff_eq!(lambda, lambda_new, epsilon = 0.0000001);

        let rc1 = risk_contribution(
            &loan1,
            &el_vec,
            &systemic_expectation,
            &var_vec,
            &v,
            lambda0,
            lambda,
            q,
            c,
        );
        let rc2 = risk_contribution(
            &loan2,
            &el_vec,
            &systemic_expectation,
            &var_vec,
            &v,
            lambda0,
            lambda,
            q,
            c,
        );
        assert_abs_diff_eq!(rc1 + rc2, liquid_exp + c * liquid_var.sqrt(), epsilon = 0.1);
    }
    #[test]
    fn test_lambda_incremental_risk_contribution_non_homogenous_internal_function() {
        let balance1 = 1.0;
        let balance2 = 2.0;
        let pd1 = 0.05;
        let pd2 = 0.03;
        let lgd1 = 0.5;
        let lgd2 = 0.4;
        let num_loans1 = 6000.0;
        let num_loans2 = 4000.0;
        let r1 = 0.1;
        let r2 = 0.2;
        let lambda0 = 100.0; //loss in the event of a liquidity crisis
        let lambda = r1 * balance1 * num_loans1 + r2 * balance2 * num_loans2;
        let q = 0.01 / (num_loans1 * pd1 * lgd1 * balance1 + num_loans2 * pd2 * lgd2 * balance2);
        let x_min = (-num_loans1 * pd1 * lgd1 * balance1
            - num_loans2 * pd2 * lgd2 * balance2
            - lambda0
            - lambda)
            * 3.0;
        let v = vec![0.4, 0.3];
        let systemic_expectation = vec![1.0, 1.0];
        let lgd_variance = 0.2;
        let weight1_1 = vec![0.4, 0.6];
        let weight2_1 = vec![0.3, 0.7];

        let x_max = 0.0;
        let num_u: usize = 1024;
        let mut discrete_cf = EconomicCapitalAttributes::new(num_u, v.len());

        let liquid_fn = get_liquidity_risk_fn(lambda0 + lambda, q);

        //the exponent is negative because l represents a loss
        let lgd_fn = |u: &Complex<f64>, l: f64, lgd_v: f64| {
            cf_functions::gamma_cf(&(-u * l), 1.0 / lgd_v, lgd_v)
        };
        let u_domain: Vec<Complex<f64>> = fang_oost::get_u_domain(num_u, x_min, x_max).collect();
        let log_lpm_cf = get_log_lpm_cf(&lgd_fn, &liquid_fn);

        let loan1 = Loan {
            pd: pd1,
            lgd: lgd1,
            balance: balance1,
            lgd_variance,
            weight: weight1_1,
            r: r1,
            num: num_loans1, //homogenous
        };
        discrete_cf.process_loan(&loan1, &u_domain, &log_lpm_cf);
        let systemic_mgf = gamma_mgf(&v);

        let loan2 = Loan {
            pd: pd2,
            lgd: lgd2,
            balance: balance2,
            lgd_variance,
            weight: weight2_1,
            r: r2,
            num: num_loans2,
        };

        let quantile = 0.01;
        let max_iterations = 100;
        let tolerance = 0.0001;
        let risk_measure_fn = |final_cf: &[Complex<f64>]| {
            let cf_dist_utils::RiskMetric{value_at_risk, ..} = cf_dist_utils::get_expected_shortfall_and_value_at_risk_discrete_cf(
                quantile,
                x_min,
                x_max,
                max_iterations,
                tolerance,
                final_cf,
            ).unwrap();
            value_at_risk
        };
        let rc1 = discrete_cf.experiment_risk_contribution(
            &loan2,
            &u_domain,
            &log_lpm_cf,
            lambda0,
            q,
            &systemic_mgf,
            &systemic_expectation,
            &v,
            &risk_measure_fn,
        );

        let EconomicCapitalAttributes {
            el_vec,
            var_vec,
            lambda: lambda_new,
            cf,
            ..
        } = discrete_cf.experiment_loan(&loan2, &u_domain, &log_lpm_cf);

        let new_variance = portfolio_variance(&el_vec, &systemic_expectation, &var_vec, &v);
        let new_expectation = portfolio_expectation(&el_vec, &systemic_expectation);
        let liquid_exp = expectation_liquidity(lambda + lambda0, q, new_expectation);
        let liquid_var = variance_liquidity(lambda + lambda0, q, new_expectation, new_variance);
        let cf_d = discrete_cf.get_experiment_full_cf(&cf, &systemic_mgf);
        let cf_dist_utils::RiskMetric{value_at_risk, ..} = cf_dist_utils::get_expected_shortfall_and_value_at_risk_discrete_cf(
            quantile,
            x_min,
            x_max,
            max_iterations,
            tolerance,
            &cf_d,
        ).unwrap();
        let c = (value_at_risk - liquid_exp) / liquid_var.sqrt();
        assert_abs_diff_eq!(lambda, lambda_new, epsilon = 0.0000001);

        let rc2 = risk_contribution(
            &loan2,
            &el_vec,
            &systemic_expectation,
            &var_vec,
            &v,
            lambda0,
            lambda,
            q,
            c,
        );

        assert_abs_diff_eq!(rc2, rc1, epsilon = 0.00001);
    }
}
