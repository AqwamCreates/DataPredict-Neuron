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
	
	print(difference())
	
	return cost
	
end

AHADeepLearningLibrary.FastMeanSquaredError = function(forwardPropagateFunction1, forwardPropagateFunction2)

	local tensor1, backwardPropagationFunction1, getTensor1 = forwardPropagateFunction1()

	local tensor2, backwardPropagationFunction2, getTensor2 = forwardPropagateFunction2()
	
	local differenceTensor
	
	local resultValue
	
	local getValue = function() 
		
		tensor1 = getTensor1()

		tensor2 = getTensor2()

		differenceTensor = AqwamTensorLibrary:subtract(tensor1, tensor2)

		local resultTensor = AqwamTensorLibrary:power(differenceTensor, 2)

		resultValue = AqwamTensorLibrary:sum(resultTensor) / (#resultTensor)
		
		return resultValue 
		
	end

	local parentBackwardPropagation = function(firstDerivativeTensor)
		
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
		
		resultValue = getValue()

		return resultValue, parentBackwardPropagation, getValue

	end

	return forwardPropagationFunction


end

return AHADeepLearningLibrary
