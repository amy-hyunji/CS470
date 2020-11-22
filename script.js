document.getElementById("submit").addEventListener("click", input);

function input() {
    var query = document.getElementById("query").value;
    var imageid = document.getElementById("target").src;
    // result = model(imageurl, query);
    result = query + " " + imageid;
    document.getElementById("output").value = result;
}