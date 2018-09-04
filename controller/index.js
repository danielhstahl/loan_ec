'use strict'
const { spawn } = require('child_process')
const numSend=1000
const batchSize=1000
const numLoans=numSend*batchSize
const numMacroWeights=3
const minLoanSize=10000
const maxLoanSize=50000
const maxP=.09
const minP=.0001
const maxPossibleLoss=.14//more than this an we are in big trouble...
const roughTotalExposure=(minLoanSize, maxLoanSize, numLoans)=>numLoans*(minLoanSize+.5*(maxLoanSize-minLoanSize))
const generateWeights=numWeights=>{
    let myWeights=[]
    for(let i=0; i<numWeights; ++i){
        myWeights.push(Math.random())
    }
    const total=myWeights.reduce((cum, curr)=>cum+curr)
    return myWeights.map(val=>val/total)
}
const generateRandom=(min, max)=>{
    return Math.random()*(max-min)+min;
}
const generateFakeLoanData=(numLoans, numMacroWeight)=>{
    let loans=[]
    for(let i=0; i<numLoans;++i){
        loans.push({
            weight:generateWeights(numMacroWeight),
            balance:generateRandom(minLoanSize, maxLoanSize),
            lgd:0.5,
            pd:generateRandom(minP, maxP)
        })
    }
    return loans
}
const genericSpawn=(binaryName, options)=>new Promise((resolve, reject)=>{
    const binSubPath=`target/release/${binaryName}`
    const binaryPath=process.env['LAMBDA_TASK_ROOT']?
      `${process.cwd()}/${binSubPath}`:
      `./${binSubPath}`
    const model=spawn(binaryPath,options)
    let modelOutput=''
    let modelErr=''
    model.stdout.on('data', data=>{
      modelOutput+=data
    })
    model.stderr.on('data', data=>{
      modelErr+=data
    })
    model.on('close', code=>{
      if(modelErr){
        return reject(modelErr)
      }
      resolve(modelOutput)
    })
})

const totalExposure=roughTotalExposure(minLoanSize, maxLoanSize, numLoans)
const parameters={
    lambda:.2*totalExposure,
    q:.1/totalExposure,
    alpha_l:.2,
    b_l:.5,
    sig_l:.2,
    t:1.0,
    u_steps:256,
    num_send:batchSize,
    x_min:-maxPossibleLoss*totalExposure, 
    x_max:0
}
const spawnOne=()=>genericSpawn('loan_cf', [
    JSON.stringify(parameters), 
    JSON.stringify(generateFakeLoanData(numLoans, numMacroWeights))
])
const spawnMany=()=>{
    let v=[]
    for(let i=0; i<numSend;++i){
        v.push(i)
    }
    Promise.all(v.map(spawnOne))
}

spawnOne()

