function sendKeyToAllTabs(keyStr) {
    chrome.windows.getAll({'populate':true}, function(windows) {
        for (let i=0; i<windows.length; i++) {
            let tabs = windows[i].tabs;
            for (let j=0; j<tabs.length; j++) {
                chrome.tabs.sendRequest(tabs[j].id, {'key':keyStr});
            }
        }
    });
}

function loadContentScriptInAllTabs() {
    chrome.windows.getAll({'populate':true}, function(windows) {
        for (let i=0; i<windows.length; i++) {
            let tabs = windows[i].tabs;
            for (let j=0; j<tabs.length; j++) {
                chrome.tabs.executeScript(tabs[j].id, {file:'keycodes.js', allFrames:true});
                chrome.tabs.executeScript(tabs[j].id, {file:'content_script.js', allFrames:true});
            }
        }
    });
}