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

`cargo build --release`

`./target/release/loan_cf`

## Roadmap

Goals (decreasing order of importance):

* Ingest loan level parameters (balance, pd, lgd, ...etc) to create a loss characteristic function
* High memory efficiency (don't store much data in ram)
* Parallelizable given a batch of loans
* Parallelizable over batches of loans (ie, make batches of loans independent of each other)

Success Criteria

* 1,000,000 loans in under 5 seconds on  i5-5250U CPU @ 1.60GHz × 4 
* Reproducable (same output given same input)
* Transparent (easy to replicate)
