var speakKeyStr;

function speakSelection() {
    let focused = document.activeElement;
    let selectedText;
    if (focused) {
        try {
            selectedText = focused.value.substring(focused.selectionStart, focused.selectionEnd);
        } catch (err) {
        }
    }
    if (selectedText == undefined) {
        let sel = window.getSelection();
        selectedText = sel.toString();
    }

    chrome.extension.sendRequest({'speak' : selectedText}); 
}

function onExtensionMessage(request) {
    if (request['key'] != undefined) {
        speakKeyStr = request['key'];
    }
}

function initContentScript() {
    chrome.extension.onRequest.addListener(onExtensionMessage);
    chrome.extension.sendRequest({'init' : true}, onExtensionMessage);

    document.addEventListener('keydown', function(evt) {
        if (!document.hasFocus()) {
            return true;
        }
        let keyStr = keyEventToString(evt);
        if (keyStr == speakKeyStr && speakKeyStr.length > 0) {
            speakSelection();
            evt.stopPropagation();
            evt.preventDefault();
            return false;
        }
        return true;
    }, false);
}

initContentScript();