import { InferenceSession, Tensor } from "onnxruntime-web";

import { softmax, imagenetClassesTopK } from './utils/modelHelper.jsx';

import { useEffect } from 'react';

function App() {

    async function test() {

        const model = `./model/classify_medicine_34_64.onnx`;

        const session = await InferenceSession.create(
            model, { executionProviders: ["webgl"], }
        );

        const input = new Tensor('float32', new Float32Array(640 * 640 * 3), [1, 3, 640, 640])

        const feeds = {};
        feeds[session.inputNames[0]] = input


        // Run the session inference.
        const outputData = await session.run(feeds);

        // Get output results with the output name from the model export.
        const output = outputData[session.outputNames[0]];

        //Get the softmax of the output data. The softmax transforms values to be between 0 and 1
        const outputSoftmax = softmax(Array.prototype.slice.call(output.data));

        //Get the top 5 results.
        const results = imagenetClassesTopK(outputSoftmax, 5);
        console.log('results: ', results);
        
    }

    useEffect(() => {
        
        test();

    }, []);


    return (
        <h1>hello world</h1>
    )
}


export default App;