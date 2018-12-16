| [Linux][lin-link] | [Codecov][cov-link] |
| :---------------: | :-----------------: |
| ![lin-badge]      | ![cov-badge]        |

[lin-badge]: https://travis-ci.com/phillyfan1138/credit_faas_demo.svg "Travis build status"
[lin-link]:  https://travis-ci.com/phillyfan1138/credit_faas_demo "Travis build status"
[cov-badge]: https://codecov.io/gh/phillyfan1138/credit_faas_demo/branch/master/graph/badge.svg
[cov-link]:  https://codecov.io/gh/phillyfan1138/credit_faas_demo

## Demo for Credit FaaS

This project showcases how to use Fang Oosterlee's algorithm for efficiently computing portfolio statistics at a granular (loan) level.  

## Documentation

Model documentation and theory is available in the [Credit Risk Extensions](https://github.com/phillyfan1138/CreditRiskExtensions/blob/master/StahlMultiVariatePaper.pdf) repository.

## Features

The model allows for:
* Correlated PD
* Stochastic LGD (but uncorrelated with anything else)
* Granularity: no need for the assumption of "sufficiently diversified" 
* Efficiency: near real-time computation

## How to build
First, download this repo:
`git clone https://github.com/phillyfan1138/credit_faas_demo`

Change directory into the folder:
`cd credit_faas_demo`

This repo contains test files that are quite large.  I use [git-lfs](https://git-lfs.github.com/) to handle these files.  If you already have git-lfs installed, the files should be downloaded automatically.  If you cloned the repo without git-lfs installed, after installation run:
`git-lfs fetch`

Regardless of whether you download the test files, you can build the binaries with:
`cargo build --release`

## How to run
To get the expected shortfall and value at risk for granular (1 million) loans, run the following:

`./target/release/loan_cf $(cat ./data/parameters.json)  ./data/loans.json`

With optional density export:

`./target/release/loan_cf $(cat ./data/parameters.json)  ./data/loans.json ./docs/loan_density_full.json`

## Recommended implementation
In a real production setting, there will typically be a finite set of segments that describe loans.  For example, a loan may have one of 10 risk ratings and one of 10 facility grades.  Loans may also be grouped by rough exposure amount (eg, roughly 10 seperate exposures).  This leads to 1000 different combinations.  Instead of simulating over every single loan, the model could simulate over each group, with each group multiplied by the number of loans in each group.  If there are 30 loans with risk rating 4, facility grade 6, and in exposure segment 5, then the exponent of the characteristic function would be 30*p(e^{uil}-1) where p is the probability of default associated with risk rating 4 and l is the combined dollar loss for a loan in segment 5 and facility grade 6.  

This will dramatically decrease the computation time.

To run the demo for this recommended implementation, 

`./target/release/loan_cf $(cat ./data/parameters.json)  ./data/loans_grouped.json`

With optional density export:

`./target/release/loan_cf $(cat ./data/parameters.json)  ./data/loans_grouped.json ./docs/loan_density_aggr.json`


## Comparison of granular and recommended implementations

Note that these plots were generated using two different simulations and are not intended to represent the same loan portfolio.  The differences when applied to a real loan portfolio should be minimal.

![](docs/density_compare.jpg?raw=true)

## Comparison with riskmetrics

[Risk Metrics](https://www.msci.com/documents/10199/93396227-d449-4229-9143-24a94dab122f) is a competing method of estimating the loss distribution of a portfolio of loans.  For a simple single factor model, the following plot shows the mapping between the systemic correlation parameter (rho) for Risk Metrics and the standard deviation of the systemic variable for Credit Risk Plus (standard_deviation):

![](docs/vol_corr_compare.jpg?raw=true)

Note that the volatility of the underlying systemic variable for Credit Risk Plus has to be relatively large to map to the same correlation for Risk Metrics.


## Roadmap

* Parsimonious integration with correlations from Risk Metrics
* Move to Gamma factor variables
    * The required standard deviation for factor variables would make Gaussian factors have a relatively high probability of being negative.
    * The Vasicek assumptions require estimation not only of the drift and volatility, but also the initial values of the process.  These are nearly impossible to estimate.

## Completed

Goals (decreasing order of importance):

* Ingest loan level parameters (balance, pd, lgd, ...etc) to create a loss characteristic function 
* High memory efficiency (don't store much data in ram)

Success Criteria

* 1,000,000 loans in under 5 seconds on  i5-5250U CPU @ 1.60GHz × 4 (done after aggregations)
* Reproducible (same output given same input) (done)
* Transparent (easy to replicate) (done)
