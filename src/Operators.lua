local AqwamTensorLibrary = require(script.Parent.AqwamTensorLibraryLinker.Value)

local Operators = {}

local function collapseTensor(tensor, targetDimensionSizeArray)

	local numberOfDimensionsOfTensor = #targetDimensionSizeArray

	local numberOfDimensionsOfDerivativeTensor = #AqwamTensorLibrary:getDimensionSizeArray(tensor)

	local numberOfDimensionsToSum = numberOfDimensionsOfDerivativeTensor - numberOfDimensionsOfTensor

	for i = 1, numberOfDimensionsToSum, 1 do tensor = AqwamTensorLibrary:sum(tensor, 1)[1] end

	for i, size in ipairs(targetDimensionSizeArray) do

		if (size == 1) then tensor = AqwamTensorLibrary:sum(tensor, i) end

	end

	return tensor

end

Operators.Input = function(tensor)
	
	local getTensor = function() return tensor end
	
	local forwardPropagationFunction = function() return tensor, nil, getTensor end

	return forwardPropagationFunction
	
end

Operators.InputToDescend = function(tensor, learningRate, Optimizer, Regularizer)
	
	local getTensor = function() return tensor end
	
	-- Can use DataPredict Axon Optimizer and Regularizer here.
	
	local backwardPropagationFunction = function(firstDerivativeTensor)
		
		if (Regularizer) then

			local regularizationTensor = Regularizer:calculate{tensor}

			firstDerivativeTensor = AqwamTensorLibrary:add(firstDerivativeTensor, regularizationTensor)

		end
		
		if (Optimizer) then
			
			firstDerivativeTensor = Optimizer:calculate{learningRate, firstDerivativeTensor}
			
		else
			
			firstDerivativeTensor = AqwamTensorLibrary:multiply(learningRate, firstDerivativeTensor)
			
		end
		
		tensor = AqwamTensorLibrary:subtract(tensor, firstDerivativeTensor)

	end

	local forwardPropagationFunction = function() 

		return tensor, backwardPropagationFunction, getTensor

	end
	
	return forwardPropagationFunction, getTensor
	
end

Operators.Add = function(forwardPropagateFunction1, forwardPropagateFunction2)
	
	local resultTensor
	
	local getTensor = function() return resultTensor end
	
	local tensor1, backwardPropagationFunction1, getTensor1 = forwardPropagateFunction1()

	local tensor2, backwardPropagationFunction2, getTensor2 = forwardPropagateFunction2()

	local parentBackwardPropagation = function(firstDerivativeTensor)

		if (backwardPropagationFunction1) then backwardPropagationFunction1(firstDerivativeTensor) end

		if (backwardPropagationFunction2) then backwardPropagationFunction2(firstDerivativeTensor) end

	end

	local forwardPropagationFunction = function()
		
		-- Lazy evaluation on forward call! Avoids producing non-dynamic outputs by only calling it inside here.
		
		tensor1 = getTensor1()

		tensor2 = getTensor2()
		
		resultTensor = AqwamTensorLibrary:add(tensor1, tensor2)

		return resultTensor, parentBackwardPropagation, getTensor

	end

	return forwardPropagationFunction
	
	
end

Operators.Subtract = function(forwardPropagateFunction1, forwardPropagateFunction2)
	
	local resultTensor

	local getTensor = function() return resultTensor end

	local tensor1, backwardPropagationFunction1, getTensor1 = forwardPropagateFunction1()

	local tensor2, backwardPropagationFunction2, getTensor2 = forwardPropagateFunction2()

	local parentBackwardPropagation = function(firstDerivativeTensor)

		if (backwardPropagationFunction1) then
			
			local targetDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(tensor1)
			
			local collapsedFirstDerivativeTensor = collapseTensor(firstDerivativeTensor, targetDimensionSizeArray)
			
			backwardPropagationFunction1(collapsedFirstDerivativeTensor) 
			
		end

		if (backwardPropagationFunction2) then
			
			local targetDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(tensor2)

			local collapsedFirstDerivativeTensor = collapseTensor(firstDerivativeTensor, targetDimensionSizeArray)
			
			backwardPropagationFunction2(AqwamTensorLibrary:unaryMinus(collapsedFirstDerivativeTensor)) 
			
		end

	end

	local forwardPropagationFunction = function() 
		
		tensor1 = getTensor1()

		tensor2 = getTensor2()
		
		resultTensor = AqwamTensorLibrary:subtract(tensor1, tensor2)

		return resultTensor, parentBackwardPropagation, getTensor

	end

	return forwardPropagationFunction


end

Operators.Multiply = function(forwardPropagateFunction1, forwardPropagateFunction2)
	
	local resultTensor

	local getTensor = function() return resultTensor end
	
	local tensor1, backwardPropagationFunction1, getTensor1 = forwardPropagateFunction1()
	
	local tensor2, backwardPropagationFunction2, getTensor2 = forwardPropagateFunction2()
	
	local parentBackwardPropagation = function(firstDerivativeTensor)
		
		if (backwardPropagationFunction1) then
			
			local targetDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(tensor1)

			local collapsedFirstDerivativeTensor = collapseTensor(firstDerivativeTensor, targetDimensionSizeArray)
			
			local chainedFirstDerivativeTensor = AqwamTensorLibrary:multiply(collapsedFirstDerivativeTensor, tensor2)
			
			backwardPropagationFunction1(chainedFirstDerivativeTensor) 
			
		end
		
		if (backwardPropagationFunction2) then
			
			local targetDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(tensor2)

			local collapsedFirstDerivativeTensor = collapseTensor(firstDerivativeTensor, targetDimensionSizeArray)
			
			local chainedFirstDerivativeTensor = AqwamTensorLibrary:multiply(collapsedFirstDerivativeTensor, tensor1)
			
			backwardPropagationFunction2(chainedFirstDerivativeTensor) 
			
		end
		
	end
	
	local forwardPropagationFunction = function()
		
		tensor1 = getTensor1()

		tensor2 = getTensor2()
		
		resultTensor = AqwamTensorLibrary:multiply(tensor1, tensor2)
		
		return resultTensor, parentBackwardPropagation, getTensor
		
	end
	
	return forwardPropagationFunction
	
end

Operators.Divide = function(forwardPropagateFunction1, forwardPropagateFunction2)

	local resultTensor

	local getTensor = function() return resultTensor end

	local tensor1, backwardPropagationFunction1, getTensor1 = forwardPropagateFunction1()

	local tensor2, backwardPropagationFunction2, getTensor2 = forwardPropagateFunction2()

	local parentBackwardPropagation = function(firstDerivativeTensor)

		if (backwardPropagationFunction1) then

			local targetDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(tensor1)

			local collapsedFirstDerivativeTensor = collapseTensor(firstDerivativeTensor, targetDimensionSizeArray)

			local chainedFirstDerivativeTensor = AqwamTensorLibrary:divide(collapsedFirstDerivativeTensor, tensor2)

			backwardPropagationFunction1(chainedFirstDerivativeTensor) 

		end

		if (backwardPropagationFunction2) then

			local targetDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(tensor2)
			
			local squaredDivisorTensor = AqwamTensorLibrary:power(tensor2, 2)
			
			local numeratorTensor = AqwamTensorLibrary:multiply(firstDerivativeTensor, tensor1, -1)
			
			local collapsedNumeratorTensor = collapseTensor(numeratorTensor, targetDimensionSizeArray)
			
			local chainedFirstDerivativeTensor = AqwamTensorLibrary:divide(collapsedNumeratorTensor, squaredDivisorTensor)

			backwardPropagationFunction2(chainedFirstDerivativeTensor) 

		end

	end

	local forwardPropagationFunction = function()

		tensor1 = getTensor1()

		tensor2 = getTensor2()

		resultTensor = AqwamTensorLibrary:multiply(tensor1, tensor2)

		return resultTensor, parentBackwardPropagation, getTensor

	end

	return forwardPropagationFunction

end

Operators.Power = function(baseForwardPropagateFunction, exponentForwardPropagateFunction)
	
	local resultTensor

	local getTensor = function() return resultTensor end

	local baseTensor, baseBackwardPropagationFunction, getBaseTensor = baseForwardPropagateFunction()

	local exponentTensor, exponentBackwardPropagationFunction, getExponentTensor = exponentForwardPropagateFunction()

	local parentBackwardPropagation = function(firstDerivativeTensor)

		if (baseBackwardPropagationFunction) then
			
			local subtractedExponentTensor = AqwamTensorLibrary:subtract(exponentTensor, 1)
			
			local partialFirstDerivativeTensor = AqwamTensorLibrary:power(exponentTensor, subtractedExponentTensor)

			local chainedFirstDerivativeTensor = AqwamTensorLibrary:multiply(firstDerivativeTensor, partialFirstDerivativeTensor)
			
			local targetDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(baseTensor)

			local collapsedFirstDerivativeTensor = collapseTensor(firstDerivativeTensor, targetDimensionSizeArray)

			baseBackwardPropagationFunction(collapsedFirstDerivativeTensor) 

		end

		if (exponentBackwardPropagationFunction) then

			local partialFirstDerivativeTensor = AqwamTensorLibrary:applyFunction(function(base, exponent) return (math.pow(base, exponent) * math.log(base)) end, baseTensor, exponentTensor)

			local collapsedChainRuleFirstDerivativeTensor = AqwamTensorLibrary:multiply(partialFirstDerivativeTensor, firstDerivativeTensor)
			
			local targetDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(exponentTensor)
			
			local collapsedFirstDerivativeTensor = collapseTensor(collapsedChainRuleFirstDerivativeTensor, targetDimensionSizeArray)

			exponentBackwardPropagationFunction(collapsedFirstDerivativeTensor) 

		end

	end

	local forwardPropagationFunction = function()

		baseTensor = getBaseTensor()

		exponentTensor = getExponentTensor()
		
		resultTensor = AqwamTensorLibrary:power(baseTensor, exponentTensor)

		return resultTensor, parentBackwardPropagation, getTensor

	end

	return forwardPropagationFunction

end

Operators.DotProduct = function(forwardPropagateFunction1, forwardPropagateFunction2)
	
	local resultTensor

	local getTensor = function() return resultTensor end
	
	local tensor1, backwardPropagationFunction1, getTensor1 = forwardPropagateFunction1()

	local tensor2, backwardPropagationFunction2, getTensor2 = forwardPropagateFunction2()
	
	local parentBackwardPropagation = function(firstDerivativeTensor)

		if (backwardPropagationFunction1) then
			
			local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(tensor2)
			
			local numberOfDimensions = #dimensionSizeArray
			
			local transposedTensor2 = AqwamTensorLibrary:transpose(tensor2, {(numberOfDimensions - 1), numberOfDimensions})
			
			local chainedFirstDerivativeTensor = AqwamTensorLibrary:dotProduct(firstDerivativeTensor, transposedTensor2)
			
			backwardPropagationFunction1(chainedFirstDerivativeTensor) 
			
		end

		if (backwardPropagationFunction2) then 
			
			local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(tensor1)

			local numberOfDimensions = #dimensionSizeArray

			local transposedTensor1 = AqwamTensorLibrary:transpose(tensor1, {(numberOfDimensions - 1), numberOfDimensions})
			
			local chainedFirstDerivativeTensor = AqwamTensorLibrary:dotProduct(transposedTensor1, firstDerivativeTensor)
			
			backwardPropagationFunction2(chainedFirstDerivativeTensor) 
			
		end

	end

	local forwardPropagationFunction = function() 
		
		tensor1 = getTensor1()

		tensor2 = getTensor2()
		
		resultTensor = AqwamTensorLibrary:dotProduct(tensor1, tensor2)

		return resultTensor, parentBackwardPropagation, getTensor

	end

	return forwardPropagationFunction
	
end

Operators.Exponent = function(forwardPropagateFunction)
	
	local resultTensor

	local getParentTensor = function() return resultTensor end
	
	local tensor, backwardPropagationFunction, getChildTensor = forwardPropagateFunction()
	
	local parentBackwardPropagation = function(firstDerivativeTensor)

		if (backwardPropagationFunction) then

			local chainedFirstDerivativeTensor = AqwamTensorLibrary:multiply(AqwamTensorLibrary:exponent(tensor), firstDerivativeTensor)

			backwardPropagationFunction(chainedFirstDerivativeTensor) 

		end

	end

	local forwardPropagationFunction = function() 

		tensor = getChildTensor()
		
		resultTensor = AqwamTensorLibrary:exponent(tensor)

		return resultTensor, parentBackwardPropagation, getParentTensor

	end

	return forwardPropagationFunction
	
end

Operators.UnaryMinus = function(forwardPropagateFunction)
	
	local resultTensor

	local getParentTensor = function() return resultTensor end

	local tensor, backwardPropagationFunction, getChildTensor = forwardPropagateFunction()

	local parentBackwardPropagation = function(firstDerivativeTensor)

		if (backwardPropagationFunction) then backwardPropagationFunction(AqwamTensorLibrary:unaryMinus(firstDerivativeTensor)) end

	end

	local forwardPropagationFunction = function() 

		tensor = getParentTensor()
		
		resultTensor = AqwamTensorLibrary:unaryMinus(tensor)

		return resultTensor, parentBackwardPropagation, getChildTensor

	end

	return forwardPropagationFunction

end

Operators.Sum = function(forwardPropagateFunction, dimension)
	
	local resultTensor
	
	local getParentTensor = function() return resultTensor end

	local tensor, backwardPropagationFunction, getChildTensor = forwardPropagateFunction()

	local parentBackwardPropagation = function(firstDerivativeTensor)

		if (backwardPropagationFunction) then backwardPropagationFunction(firstDerivativeTensor) end

	end

	local forwardPropagationFunction = function() 
		
		tensor = getChildTensor()
		
		resultTensor = AqwamTensorLibrary:sum(tensor, dimension)

		return resultTensor, parentBackwardPropagation, getParentTensor

	end

	return forwardPropagationFunction

end

Operators.Logarithm = function(numberForwardPropagateFunction, baseForwardPropagateFunction)
	
	local numberTensor, numberBackwardPropagationFunction, getNumberTensor = numberForwardPropagateFunction()

	local baseTensor, baseBackwardPropagationFunction, getBaseTensor
	
	if (baseForwardPropagateFunction) then
		
		baseTensor, baseBackwardPropagationFunction, getBaseTensor = baseForwardPropagateFunction()
		
	end
	
	local parentBackwardPropagation = function(firstDerivativeTensor)

		if (numberBackwardPropagationFunction) then
			
			local partialFirstDerivativeFunctionToApply

			local partialFirstDerivativeTensor

			if (baseTensor) then

				partialFirstDerivativeFunctionToApply = function (number, base) return (1 / (number * math.log(base))) end

				partialFirstDerivativeTensor = AqwamTensorLibrary:applyFunction(partialFirstDerivativeFunctionToApply, numberTensor, baseTensor)

			else

				partialFirstDerivativeFunctionToApply = function (number) return (1 / number) end

				partialFirstDerivativeTensor = AqwamTensorLibrary:applyFunction(partialFirstDerivativeFunctionToApply, numberTensor)

			end
			
			local chainedFirstDerivativeTensor = AqwamTensorLibrary:multiply(firstDerivativeTensor, partialFirstDerivativeTensor)
			
			local numberTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(numberTensor)

			local collapsedFirstDerivativeTensor = collapseTensor(chainedFirstDerivativeTensor, numberTensorDimensionSizeArray)
			
			numberBackwardPropagationFunction(collapsedFirstDerivativeTensor)

		end

		if (baseBackwardPropagationFunction) then

			local partialFirstDerivativeFunctionToApply = function (number, base) return -(math.log(number) / (base * math.pow(math.log(base), 2))) end

			local partialFirstDerivativeTensor = AqwamTensorLibrary:applyFunction(partialFirstDerivativeFunctionToApply, numberTensor, baseTensor)
			
			local chainedFirstDerivativeTensor = AqwamTensorLibrary:multiply(firstDerivativeTensor, partialFirstDerivativeTensor)
			
			local baseTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(baseTensor)
			
			local collapsedFirstDerivativeTensor = collapseTensor(chainedFirstDerivativeTensor, baseTensorDimensionSizeArray)
			
			baseBackwardPropagationFunction(collapsedFirstDerivativeTensor)

		end

	end
	
	local forwardPropagationFunction = function() 

		numberTensor = getNumberTensor()
		
		baseTensor = getBaseTensor()

		return AqwamTensorLibrary:logarithm(numberTensor, baseTensor), parentBackwardPropagation, getTensor

	end

	return forwardPropagationFunction
	
end

return Operators
