async function runExample() {
    // Create an array of float numbers from the input fields
    var x = [
        parseFloat(document.getElementById('box1').value),
        parseFloat(document.getElementById('box2').value),
        parseFloat(document.getElementById('box3').value),
        parseFloat(document.getElementById('box4').value),
        parseFloat(document.getElementById('box5').value),
    ];

    // Create a tensor from the array
    let tensorX = new onnx.Tensor(new Float32Array(x), 'float32', [1, 5]);

    // Create a new inference session and load the model
    let session = new onnx.InferenceSession();
    await session.loadModel("./Pitch_Type_Model.onnx");

    // Run the model
    let outputMap = await session.run({input: tensorX});
    let outputData = outputMap.get('output1');

    // Update the predictions div
    let predictions = document.getElementById('predictions');
    predictions.innerHTML = `
    <hr> Got an output tensor with values: <br/>
    <table>
        <tr>
            <td> Pitch Type Prediction </td>
            <td id="td0"> ${outputData.data[0].toFixed(2)} </td>
        </tr>
    </table>`;
}
