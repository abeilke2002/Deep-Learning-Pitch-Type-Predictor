async function runExample() {
    
    var x = [
        parseFloat(document.getElementById('box1').value),
        parseFloat(document.getElementById('box2').value),
        parseFloat(document.getElementById('box3').value),
        parseFloat(document.getElementById('box4').value),
        parseFloat(document.getElementById('box5').value),
    ];

    let tensorX = new onnx.Tensor(x, 'float32', [1, 5]);

    let session = new onnx.InferenceSession();
    await session.loadModel("./Pitch_Type_Model.onnx");
    let outputMap = await session.run({input: tensorX});
    let outputData = outputMap.get('output1').data;

    let predictedClassIndex = outputData.findIndex(value => value === Math.max(...outputData));

    let predictions = document.getElementById('predictions');
    predictions.innerHTML = ` <hr> Got an output tensor with values: <br/>
        <table>
            <tr>
                <td> Pitch Type Prediction </td>
                <td id="td0"> ${predictedClassIndex} </td>
            </tr>
        </table>`;
}
