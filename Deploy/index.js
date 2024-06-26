async function runExample() {

    var x = new Float32Array( 1, 5 )

    var x = [];

     x[0] = document.getElementById('box1').value;
     x[1] = document.getElementById('box2').value;
     x[2] = document.getElementById('box3').value;
     x[3] = document.getElementById('box4').value;
     x[4] = document.getElementById('box5').value;

    let tensorX = new onnx.Tensor(x, 'float32', [1, 5]);

    let session = new onnx.InferenceSession();

    await session.loadModel("./Pitch_Type_Model.onnx");
    let outputMap = await session.run([tensorX]);
    let outputData = outputMap.get('output1');

   let predictions = document.getElementById('predictions');

  predictions.innerHTML = ` <hr> Got an output tensor with values: <br/>
   <table>
     <tr>
       <td>  Pitch Type Prediction  </td>
       <td id="td0">  ${Math.round(outputData.data[0])}  </td>
     </tr>
  </table>`;
    


}
