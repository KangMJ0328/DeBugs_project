
const toggleBtn = document.querySelector('.nav_hamberger'); 
const menu = document.querySelector('.nav_list');
const icon = document.querySelector('.fa-bars'); /* 버거모양 아이콘 */
const icon2 = document.querySelector('.fa-times'); /* 엑스모양 아이콘 */

toggleBtn.addEventListener('click',() => {
	menu.classList.toggle('active');
	icon.classList.toggle('active');
	icon2.classList.toggle('active');
});


/* 
document : 브라우저에서 제공하는 객체  
querySelector('선택자')  : 선택자에 html태그를 입력하면  해당태그를 선택할  수 있음
querySelectorAll('선택자','선택자')  : 태그 여러개 선택 가능

*/

/* 
addEventListener ('이벤트')
이벤트를 등록하는 방법 

*/





/* <변수선언> 
 const 변하지 않는 변수명을 선언할때
 let 변할 수 있는 변수명을 선언할때  */

/* alert 알려줌 - 메세지 띄우기 (확인버튼)
   prompt 입력받음 
   confirm 확인받음 - 확인버튼,취소버튼 */




/* javascript함수는 실행되기 전 모든 위치의 변수를 초기에 선언해 놓음
하지만, 함수표현식으로 작성하면 순서대로 코드를 읽어 코드에 도달하면 생성함

함수 선언식 --> 
	function sayHello() {
		console.log('Hello');
	}	

함수표현식 --> 
	let sayHello = function(){
		console.log('Hello');
	}

*/

/* <화살표 함수>
함수표현식 --- 
	let add = funtion(num1, num2){
		return num1 + num2;}
화살표함수 ---
	let add = (num1, num2)=>{
		return num1 + num2;}

	let add = (num1, num2)=>(
		num1 + num2;)             화살표 함수는 리턴문을 없애고 일반 괄호로 변경 가능, 리턴문이 한줄일 경우 괄호 생략도 가능
	 
	let add = num1 =>num1 + num2; 인수가 하나라면 인수에 있는 괄호도 생략 가능, 인수가 없을땐 괄호 생략 불가능!

*/
	 


















