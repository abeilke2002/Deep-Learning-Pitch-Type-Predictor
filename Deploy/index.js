async function runExample() {
    try {
        var x = [];
        x[0] = parseFloat(document.getElementById('box1').value);
        x[1] = parseFloat(document.getElementById('box2').value);
        x[2] = parseFloat(document.getElementById('box3').value);
        x[3] = parseFloat(document.getElementById('box4').value);
        x[4] = parseFloat(document.getElementById('box5').value);

        let tensorX = new onnx.Tensor(new Float32Array(x), 'float32', [1, 5]);

        let session = new onnx.InferenceSession();
        await session.loadModel("./Pitch_Type_Model_2.onnx");
        let outputMap = await session.run({ input: tensorX });
        let outputData = outputMap.get('output1');

        console.log('Output Data:', outputData.data);

        let predictions = document.getElementById('predictions');
        predictions.innerHTML = `<hr> Got an output tensor with values: <br/>
        <table>
            <tr>
                <td>Pitch Type Prediction</td>
                <td id="td0">${Math.round(outputData.data[0])}</td>
            </tr>
        </table>`;
    } catch (error) {
        console.error('Error running model:', error);
    }
}
