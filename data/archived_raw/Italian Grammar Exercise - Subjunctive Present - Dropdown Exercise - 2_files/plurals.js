setUpPage();
function setUpPage(){
	for(suindex = 0;suindex<questionArray.length;suindex++){
		document.write("<div class='exercise_row'><span class='exercise_number'>" + (suindex+1) + " </span><span class='exercise_text'>");
		document.write(questionArray[suindex][0]);
		document.write("</span><span class='exercise_gapfillbox'><input size=" + questionArray[suindex][3] + " name=ans" + (suindex+1) + "> </span><span class='exercise_text'>");
              		document.write(questionArray[suindex][2]);
            		document.write("</span><span class='exercise_marking'><img src='bits/transparent.gif' name='im" + (suindex +1) + "'></span> </div>");
		
		}
		{
			document.write("<div class='exercise_buttons'><input onClick=checkScore() type=button value='Punteggio' name='button'> <input type=button value='Soluzioni' onClick='reveal()' name='button2'> <input type='reset' onClick='again()' name='reset' value='Riprova'></div>");
			}
		
}

var answerArray = new Array(document.form1.ans1,document.form1.ans2,document.form1.ans3,document.form1.ans4,document.form1.ans5,document.form1.ans6,document.form1.ans7,document.form1.ans8,document.form1.ans9,document.form1.ans10);
var imageArray = new Array(document.images.im1,document.images.im2,document.images.im3,document.images.im4,document.images.im5,document.images.im6,document.images.im7,document.images.im8,document.images.im9,document.images.im10);
function checkScore() {
var answerArray = new Array(document.form1.ans1,document.form1.ans2,document.form1.ans3,document.form1.ans4,document.form1.ans5,document.form1.ans6, document.form1.ans7,document.form1.ans8,document.form1.ans9,document.form1.ans10);
var imageArray = new Array(document.images.im1,document.images.im2,document.images.im3,document.images.im4,document.images.im5,document.images.im6, document.images.im7,document.images.im8,document.images.im9,document.images.im10);
	if(cheat){
		alert("Per favore premi Riprova per riprovare")
	}else{
		var score = 0;
		for(csindex = 0;csindex<questionArray.length;csindex++){
			if (answerArray[csindex].value == questionArray[csindex][1]) {
				imageArray[csindex].src="bits/tick.gif";
				score++;
			} else {
				imageArray[csindex].src="bits/cross.gif";
			}
		}
		alert("Il tuo punteggio e "+score+" / " + questionArray.length);
	}
}




function reveal() {
var answerArray = new Array(document.form1.ans1,document.form1.ans2,document.form1.ans3,document.form1.ans4,document.form1.ans5,document.form1.ans6, document.form1.ans7,document.form1.ans8,document.form1.ans9,document.form1.ans10);
var imageArray = new Array(document.images.im1,document.images.im2,document.images.im3,document.images.im4,document.images.im5,document.images.im6, document.images.im7,document.images.im8,document.images.im9,document.images.im10);
	for(revindex = 0;revindex<questionArray.length;revindex++){
		answerArray[revindex].value = questionArray[revindex][1];
		imageArray[revindex].src="bits/transparent.gif";
	}
	cheat = true;
}
function again() {
var answerArray = new Array(document.form1.ans1,document.form1.ans2,document.form1.ans3,document.form1.ans4,document.form1.ans5,document.form1.ans6, document.form1.ans7,document.form1.ans8,document.form1.ans9,document.form1.ans10);
var imageArray = new Array(document.images.im1,document.images.im2,document.images.im3,document.images.im4,document.images.im5,document.images.im6, document.images.im7,document.images.im8,document.images.im9,document.images.im10);
	cheat = false;
	for(agindex = 0;agindex<questionArray.length;agindex++){
		answerArray[agindex].value = "";
		imageArray[agindex].src="bits/transparent.gif";
	}

}



