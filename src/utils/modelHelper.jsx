import _ from 'lodash';
import { imagenetClasses } from '../data/imagenet';


//The softmax transforms values to be between 0 and 1
export function softmax(resultArray) {
    // Get the largest value in the array.
    const largestNumber = Math.max(...resultArray);
    // Apply exponential function to each result item subtracted by the largest number, use reduce to get the previous result number and the current number to sum all the exponentials results.
    const sumOfExp = resultArray.map((resultItem) => Math.exp(resultItem - largestNumber)).reduce((prevNumber, currentNumber) => prevNumber + currentNumber);
    //Normalizes the resultArray by dividing by the sum of all exponentials; this normalization ensures that the sum of the components of the output vector is 1.
    return resultArray.map((resultValue, index) => {
      return Math.exp(resultValue - largestNumber) / sumOfExp;
    });
}


/**
 * Find top k imagenet classes
 */
export function imagenetClassesTopK(classProbabilities, k = 5) {
    const probs =
        _.isTypedArray(classProbabilities) ? Array.prototype.slice.call(classProbabilities) : classProbabilities;
  
    const sorted = _.reverse(_.sortBy(probs.map((prob, index) => [prob, index]), (probIndex ) => probIndex[0]));
  
    const topK = _.take(sorted, k).map((probIndex) => {
      const iClass = imagenetClasses[probIndex[1]];
      return {
        index: parseInt(probIndex[1].toString(), 10),
        name: iClass.replace(/_/g, ' '),
        probability: probIndex[0]
      };
    });
    return topK;
  }