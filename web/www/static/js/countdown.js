//숫자 프로토타입으로 입력 길이만큼 앞에 0을 채운 문자열 반환
Number.prototype.fillZero = function(width){
	//문자열 변환
    let n = String(this);
	//남는 길이만큼 0으로 채움
    return n.length >= width ? n:new Array(width-n.length+1).join('0')+n;
}

function refresh_data() {
	var now = new Date();

	var curt_secs = (now.getHours() * 3600) + (now.getMinutes() * 60) + now.getSeconds();

	var interval = parseInt(curt_secs / 10800);

	var remain_time = 10800 - (curt_secs % 10800);

	var remain_hour = parseInt(remain_time / 3600);
	var remain_minute = parseInt((remain_time % 3600) / 60);
	var remain_secs = remain_time % 60;

	$('#hours').text(remain_hour.fillZero(2))
	$('#minutes').text(remain_minute.fillZero(2))
	$('#seconds').text(remain_secs.fillZero(2))

	$('#interval').text(interval)

	setTimeout('refresh_data()', 1000);
}
