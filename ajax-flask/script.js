document.getElementById("query").addEventListener("click", image);
document.getElementById("submit").addEventListener("click", input);

var imageid = "";

function image() {
    chrome.tabs.query({
            'active': true,
            'windowId': chrome.windows.WINDOW_ID_CURRENT
        },
        function (tabs) {
            imageid = tabs[0].url.split("url=")[1];
            tabs[0].url.split("url=")[1];
        }
    );
}

function input() {
    var query = document.getElementById("query").value;
    // var imageid = document.getElementById("target").src;
    // result = model(imageurl, query);

    $.ajax({
	url:"http://localhost:5000/test",
	type:"POST",
	data:JSON.stringify({'id': 5}),
	dataType : 'JSON',
	contentType: "application/json",
	success: function(data) {
	    if (data) {
		document.getElementById("output").value = data.result;
	    } else {
		document.getElementById("output").value = query;
	    }
	},
	error: function() {
	    document.getElementById("output").value = query;
	}
    });
    
    //result = query + "\n" + imageid;
    //document.getElementById("output").value = result;
}
