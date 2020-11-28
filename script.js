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
    result = query + "\n" + imageid;
    document.getElementById("output").value = result;
}