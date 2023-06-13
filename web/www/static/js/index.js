
// const pageable = new Pageable("#wrap", {
// 	animation: 500,
// 	events: {
// 		mouse: false,
// 	}
// });

// $('#file-upload').click(function(){$('#video-upload').trigger('click');});

// ['dragleave', 'drop', 'dragenter', 'dragover'].forEach(function (evt) {
// 	document.addEventListener(evt, function (e) {
// 		e.preventDefault();
// 	}, false);
// });

// var drop_area = document.getElementById('drop_area');
// drop_area.addEventListener('drop', function (e) {
// 	e.preventDefault();
// 	var fileList = e.dataTransfer.files; // the files to be uploaded
// 	if (fileList.length == 0) {
// 		return false;
// 	}

// 	// we use XMLHttpRequest here instead of fetch, because with the former we can easily implement progress and speed.
// 	var xhr = new XMLHttpRequest();
// 	// xhr.open('post', '/drag_upload', true); // aussume that the url /upload handles uploading.
// 	xhr.open('post', '/drag_upload', true);
// 	xhr.onreadystatechange = function () {
// 		if (xhr.readyState == 4 && xhr.status == 200) {
// 			alert("성공");
// 			// uploading is successful
// 			Swal.fire({
// 				text: "업로드가 완료되었습니다.",
// 				confirmButtonColor: "#000000",
// 				icon: "success"
// 			}).then(function(){
// 				pageable.scrollToAnchor("#keyword");
// 				var id = xhr.responseText[0];
// 				$('#video-id').val(id);
// 			})
// 		}
// 	};

// 	// show uploading progress
// 	var lastTime = Date.now();
// 	var lastLoad = 0;
// 	xhr.upload.onprogress = function (event) {
// 		if (event.lengthComputable) {
// 			// update progress
// 			var percent = Math.floor(event.loaded / event.total * 100);
// 			document.getElementById('upload_progress').textContent = percent + '%';

// 			// update speed
// 			var curTime = Date.now();
// 			var curLoad = event.loaded;
// 			var speed = ((curLoad - lastLoad) / (curTime - lastTime) / 1024).toFixed(2);
// 			document.getElementById('speed').textContent = speed + 'MB/s'
// 			lastTime = curTime;
// 			lastLoad = curLoad;
// 		}
// 	};

// 	// send files to server
// 	xhr.setRequestHeader('X-Requested-With', 'XMLHttpRequest');
// 	var fd = new FormData();
// 	for (let file of fileList) {
// 		fd.append('files', file);
// 	}
// 	lastTime = Date.now();

// 	xhr.send(fd);
// }, false);
	


// function changeValue(obj){
// 	var filename = $('input[id=video-upload]').val().replace(/C:\\fakepath\\/i, '');
// 	$('.upload').submit();

// 	$('#video-file-name').val(filename);
// 	alert(filename)
// 	$.ajax({
// 		url:'/get_video_id',
// 		data: {
// 			'filename' : filename
// 		},
// 		type: 'POST',
// 		success: function(data){
// 			alert('성공');
// 			var json = JSON.parse(data);
// 			alert(json);
// 			alert(json.status);
// 			alert(json.video_id);
// 			Swal.fire({
// 				text: "업로드가 완료되었습니다.",
// 				confirmButtonColor: "#000000",
// 				icon: "success"
// 			}).then(function(){
// 				pageable.scrollToAnchor("#keyword");
// 				var id = data.video_id;
// 				$('#video-id').val(id);
// 			})
// 		},
// 		error: function(request, status, error){
// 			alert('ajax 통신 실패')
// 			alert(error);
// 			alert("code : " + request.status + "\n" + "message : " + request.responseText + "\n" + "error : " + error);
// 		}
// 	})
// }

// function sendKeyword(){
// 	$.ajax({
// 		url:'/keyword',
// 		contentType : 'application/json',
// 		method:'POST',
// 		data:JSON.stringify({
// 			keyword: $("#input-keyword").val(),
// 		}),
// 			dataType : 'JSON',
// 			contentType : 'application/json',
// 			success: function(data){
// 				if(data.status == '200') {
// 					alert('성공')
// 					alert(data.results[0]['object_key'])
// 					$("#key-table").append(
// 						$("<tr></tr>")
// 						.append($("<td></td>").attr("class", "text-center").text(data.results[0]['object_key']))
// 						.append($("<td></td>").attr("class", "text-center").append($("<input></input>").attr("id",cnt).attr("type","button").attr("class","del-button").attr("value","삭제")  ))
// 					)
// 				}
// 			},
// 			error: function(request, status, error){
// 				alert('ajax 통신 실패')
// 				alert(error);
// 			}

// 	}).done(function(res){
// 		$("#input-keyword").val('');
// 		window.location='/#keyword';
// 	});
// }

// //----------keyword------------
// function videoConvert(){
// 	var dataArrayToSend = [];

// 	$("#key-table tr").each(function(){
// 		var len = $(this).find("td").length;

// 		for(var i=0; i<len; i+=2){
// 			dataArrayToSend.push($(this).find("td").eq(i).text());
// 		}
// 	})

// 	var video_id = $('#video-id').val();

// 	$.ajax({
// 		url:'/keyword',
// 		dataType: 'json',
// 		data: {
// 			video_id : JSON.stringify(video_id),
// 			keyword : JSON.stringify(dataArrayToSend)
// 		},
// 		type: 'POST',
// 		success: function(data){
// 			alert('성공');
			
// 		},
// 		error: function(request, status, error){
// 			alert('ajax 통신 실패')
// 			alert(error);
			
// 		}
// 	})
// }

// //----------Drag & Drop--------------
// // prevent the default behavior of web browser


