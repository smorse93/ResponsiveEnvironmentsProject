let mousePositions = [];
const MAX_POS = 50;

function setup() {
  createCanvas(300, 400);
}

function draw() {
  background(220);
  //how you're drawing your pose
  ellipse(mouseX, mouseY, 50, 50);
  
  //how you're storing the last 50 poses
  mousePositions.push({x: mouseX, y: mouseY});
  
  //removes poses that are older than 50
  if (mousePositions.length > MAX_POS) {
  	 mousePositions.shift();
  }
  for (let i = 0; i < mousePositions.length; i +=1) {
    // how you want to draw the previous poses
    // relate it to i to change pose drawing over time
  	ellipse(mousePositions[i].x, mousePositions[i].y, i, i);
  }
}