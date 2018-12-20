//TODO!!! most of these should be in loan_ec.rs
pub fn variance_liquidity(
    lambda:f64,
    q:f64,
    expectation:f64,
    variance:f64
)->f64{
    variance*(1.0+q*lambda).powi(2)-expectation*q*lambda.powi(2)
}

pub fn expectation_liquidity(
    lambda:f64,
    q:f64,
    expectation:f64
)->f64{
    expectation*(1.0+q*lambda)
}

pub fn generic_risk_contribution(
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
            pd*(variance_l+expectation_l.powi(2))
        )*(1.0+q*lambda_0).powi(2)+
        coef*(
            2.0*rj*balance*q*variance_portfolio+
            rj*balance*q.powi(2)*variance_portfolio*(lambda+lambda_0)
        )
}

pub fn variance_from_independence(weights:&[f64], variances:&[f64])->f64{
    weights.iter().zip(variances).map(|(w, v)|{
        v*w.powi(2)
    }).sum()
}

pub fn scale_contributions(
    risk_measure:f64,
    expectation_liquid:f64,
    variance_liquid:f64
)->f64{
    (risk_measure-expectation_liquid)/variance_liquid
}



#[cfg(test)]
mod tests {
    use super::*;
    /*#[test]
    fn variance_works_for_simple_portfolios(){
        let pd=vec![0.05, 0.03, 0.06];
        let expectation_l=vec![0.5, 0.3, 0.2];
        let variance_systemic=vec![0.5, 0.4];
        let variance_l=vec![0.0, 0.0, 0.0];
        let balance=vec![1.0, 1.0, 1.0];
        let weights=vec![vec![0.5, 0.5], vec![0.2, 0.8], vec![0.3, 0.7]];
        let variance=portfolio_variance(
            &pd, &expectation_l,
            &variance_l, &balance, &weights,
            &variance_systemic
        );
        let expected_variance=
            (pd[0]*expectation_l[0]*balance[0]).powi(2)*(weights[0][0].powi(2)*variance_systemic[0]+weights[0][1].powi(2)*variance_systemic[1])+
            (pd[1]*expectation_l[1]*balance[1]).powi(2)*(weights[1][0].powi(2)*variance_systemic[0]+weights[1][1].powi(2)*variance_systemic[1])+
            (pd[2]*expectation_l[2]*balance[2]).powi(2)*(weights[2][0].powi(2)*variance_systemic[0]+weights[2][1].powi(2)*variance_systemic[1]);
            
        assert_abs_diff_eq!(variance, expected_variance, epsilon=0.0000001);

    }*/
    /*#[test]
    fn risk_contribution_sums_to_total(){
        let pd=vec![0.05, 0.03, 0.06];
        let expectation_l=vec![0.5, 0.3, 0.2];
        let variance_l=vec![0.2, 0.3, 0.4];
        let balance=vec![3.0, 2.0, 3.0];
        let variance_systemic=vec![0.5];
        let r=vec![0.3, 0.25, 0.5];
        let lambda0=1.0;
        let weights=vec![vec![1.0], vec![1.0], vec![1.0]];
        let lambda=lambda0+balance.iter().zip(&r).map(|(b, ri)|{b*ri}).sum::<f64>();
        let expectation_portfolio=portfolio_expectation(
            &pd, &expectation_l, &balance
        );
        let variance_portfolio=portfolio_variance(
            &pd, &expectation_l, &variance_l,
            &balance, &weights, &variance_systemic
        );
        let q=0.006;

        let c=5.0;//arbitrary
        let known_result=expectation_liquidity(lambda, q, expectation_portfolio)+
            variance_liquidity(lambda, q, variance_portfolio, expectation_portfolio)*c;
        let total=izip!(
            &pd, 
            &expectation_l, 
            &variance_l, 
            &r,
            &balance
        ).map(|(p, el, vl, ri, b)|{
            generic_risk_contribution(
                *p, *el, *vl, expectation_portfolio, variance_portfolio, 
                variance_systemic[0], c, *ri, *b, lambda0, lambda, q
            )
        }).sum();

        assert_abs_diff_eq!(known_result, total, epsilon=0.0000001);
    }*/
}