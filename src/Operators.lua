local AqwamTensorLibrary = require(script.Parent.AqwamTensorLibraryLinker.Value)

local Operators = require(script.Parent.Operators)

local AHADeepLearningLibrary = {}

AHADeepLearningLibrary.MeanSquaredError = function(forwardPropagateFunction1, forwardPropagateFunction2)
	
	local exponent = Operators.Input(2)
	
	local difference = Operators.Subtract(forwardPropagateFunction1, forwardPropagateFunction2)
	
	local differenceSquared = Operators.Power(difference, exponent)
	
	local sumDifferenceSquared = Operators.Sum(differenceSquared)
	
	local numberOfDataDivisor = Operators.Input(#difference())
	
	local cost = Operators.Divide(sumDifferenceSquared, numberOfDataDivisor)
	
	return cost
	
end

AHADeepLearningLibrary.FastMeanSquaredError = function(forwardPropagateFunction1, forwardPropagateFunction2)

	local tensor1, backwardPropagationFunction1, getTensor1 = forwardPropagateFunction1()

	local tensor2, backwardPropagationFunction2, getTensor2 = forwardPropagateFunction2()

	local parentBackwardPropagation = function(firstDerivativeTensor)
		
		local differenceTensor = AqwamTensorLibrary:subtract(tensor1, tensor2)
		
		local numberOfData = #differenceTensor

		if (backwardPropagationFunction1) then 
			
			local chainedFirstDerivativeTensor = AqwamTensorLibrary:multiply(firstDerivativeTensor, differenceTensor, 2)
			
			chainedFirstDerivativeTensor = AqwamTensorLibrary:divide(chainedFirstDerivativeTensor, numberOfData)
			
			backwardPropagationFunction1(chainedFirstDerivativeTensor) 
		end

		if (backwardPropagationFunction2) then 
			
			local chainedFirstDerivativeTensor = AqwamTensorLibrary:multiply(firstDerivativeTensor, differenceTensor, -2)

			chainedFirstDerivativeTensor = AqwamTensorLibrary:divide(chainedFirstDerivativeTensor, numberOfData)
			
			backwardPropagationFunction2(chainedFirstDerivativeTensor) 
			
		end

	end
	

	local forwardPropagationFunction = function() 
		
		tensor1 = getTensor1()

		tensor2 = getTensor2()
		
		local resultTensor = AqwamTensorLibrary:subtract(tensor1, tensor2)
		
		resultTensor = AqwamTensorLibrary:power(resultTensor, 2)
		
		local resultValue = AqwamTensorLibrary:sum(resultTensor) / (#resultTensor)

		return resultValue, parentBackwardPropagation

	end

	return forwardPropagationFunction


end

return AHADeepLearningLibrary
