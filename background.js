function speak(utterance) {
    let rate = localStorage['rate'] || 1.0;
    let pitch = localStorage['pitch'] || 1.0;
    let volume = localStorage['volume'] || 1.0;
    var voice = localStorage['voice'];
    chrome.tts.speak(
        utterance,
        {
            voiceName : voice,
            rate : parseFloat(rate),
            pitch : parseFloat(pitch),
            volume : parseFloat(volume)
        }
    );
}

function initBackground() {
    // Key setting and content script loading should be done
    // to all tabs including the ones that are already turned on. 
    loadContentScriptInAllTabs();
    
    let defaultKeyString = getDefaultKeyString();
    let keyString = localStorage['speakKey']
    if (keyString == undefined) {
        keyString = defaultKeyString;
        localStorage['speakKey'] = keyString;
    }
    sendKeyToAllTabs(keyString);

    chrome.extension.onRequest.addListener(
        function(request, sender, sendResponse) {
            if (request['init']) {
                sendResponse({'key' : localStorage['speakKey']});
            } else if (request['speak']) {
                speak(request['speak']);
            }
        }
    );
}

initBackground();