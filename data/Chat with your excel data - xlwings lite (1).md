# Chat with your excel data - xlwings lite

**Kokchun Giang-9:** [00:00:00] Hello and welcome to this video where we use Excel Wings Light as a Python in Excel that you will have an Python editor in Excel. And using this Excel Wings light, we will create a chat bot. So we'll create a way for us to chat with our. Own data in Excel. So we'll connect with Gemini API to do that.

So this will be super fun. So we'll put in Pokemon data. So you can see in the file here, you can see in the image here that what we'll be creating. So it'll be really fun and I hope that you will learn a lot in the way. So let's get started and move into the browser.

**Kokchun Giang-1:** Start with going into kaggle kaggle.com and then you search for Pokemon with stats, and I will leave this link in the description. Then you download this dataset, click on download, and you download dataset as sip. And in order to do this, you need [00:01:00] to be logged in and you can log in with your Google account~~ into ~~into this Kaggle.

And then you download ~~the, ~~this dataset. And once we have this dataset, we will put it into our. Excel sheet. It'll be downloaded as CSV, however it's possible to import it ~~into ~~into excel. I'll show you how. But before we exit the browser, let me show you another thing that we need.

Since ~~I will in this ~~I will use Google generative ai Google Gemini. Actually, I'll search that Gemini. And let's see. Gemini, API, Google Gemini, developer API. Go in there and here you should log into your Google account and then you go into exploring models in Google AI Studio.

And then you click on here, get API Key. And then you have your API key down here that~~ you, ~~you [00:02:00] should create an API key if you, or create, set up a project if you don't already have it. I will be using~~ generative I will use ~~Gemini the API. In order for us to work with~~ the ~~the LLM.

But you could use open AI if you would want it, or you could use whatever other L-L-M-A-P-I you want. And also it's possible to use local models, but then you need to serve them yourself using fast API. But I won't show you that. We'll keep it simple and just use this one. Okay.

And then. Now we'll move back to Excel Wings Light or Excel.

**Kokchun Giang-2:** Here I am in Excel. Let me ~~imp improve in, ~~increase this a little bit. Okay, let's create another sheet here. And here I will put in my data and here you click. I'm afraid mine is in Swedish, but~~ you, ~~it should be similar. Archive and Import. ~~import Tierra ~~Import, [00:03:00] it says CSV file, and you click on import, and then you go into.

The downloads where you downloaded your Pokemon csv or you download an archive and then you need to unzip it. Then you get this Pokemon csv. Click this one. And ~~hemp the data means ~~fetch data. Click this one and first it says~~ it's ~~it needs the limiter, you choose semicolon, tabs, et cetera.

Let's continue here next. And then you choose here what you have and CSV commas, separated values. And if you look into the file, it's actually commas. You click on commas and then it'll separate. Then you click on next and done and import. Yes. And this is called data. Here you have a lot of Pokemons.

And it's time for us to create our interface. [00:04:00] Now we have the data,

**Kokchun Giang-3:** Let's build our graphical user interface. Here we have, I'll go to, to the left here. First of all, I want to have a headline chat with your data. And this one I will make it a little bit bigger

and I will make it bold. I want it to be a. Much, much better. I actually want all of them to be a, I want to change the font avenue. Okay, perfect. And here I want to have, ask your, ask a question about your data. I want this to be somewhat bigger as well. I want this one to be even bigger. Wanted to make [00:05:00] some spacing here.

And then I want to answer on this sh this cell here B five I've chosen. B five, I will make it into thick thick lines here you can see this is the prompt. Cell that we want to put the prompt, I'll make this one a little bit bigger, for example, like this, and I'll make this one bigger that we have a prompt here, prompt cell, and we want to make some space in between like this.

And let me see, like this one. And then you can make it into middle. No, not this one. This one? Yeah. This could work, but then we need to make it a little bit, let me see. Like this maybe, oh yeah, but this one doesn't look nice. Okay. Yes, it's fine. And [00:06:00] also I want to I don't like the grid lines because my user interface, I want it to be clean.

Go into data or show, actually this ~~means show start ~~means, supporting lines or grid lines, remove that one. Ah, nice. Looks much better. Much better, but maybe a little bit harder to develop, but it's fine. Here, I'll have question and then ~~I will have my, ~~I will have an empty space here, and here I will have the bot answer.

This one, I want to make it into bold. I also want to make this somewhat bigger, something like this. Some spacing in between. And then I want here to be my~~ the ~~prompt. When I type something, ~~I want ~~I want it to jump down to here I can see my question here. And then if I type another prompt, I want it to replace this question.

Yeah, I could assign it to make all the history. [00:07:00] Keep going. But I will leave that for you that you have something fun to do as well. The bot answer, let us also make it into thick lines here and we can make it somewhat bigger. 'cause ~~we ~~the answer could be long I want to make this in the middle.

Yeah. Something like this. Now we have an interface that we can work with, can make it a little bit bigger. Something like this. Okay. And we can, let's see, can we, yeah, that, that is fine. And what we need is a button here. I'll move back to my browser and I'll go into this page here. Free icons download or it's called UX Wing.

I'll I want to have a chat. Okay, chat here. Let me choose ~~one. I want to see maybe something like this one. Maybe something like this ~~one. Yeah, this one. This one will work. Download the [00:08:00] SVG. Chat button or chat icon.

I will go back here and find chat icon. Let's see, chat icon and drag it in here. Okay, now we have our chat icon.

Make it a little bit smaller. I want to make this a little bit bigger here that ~~we have C ~~we have C five which is the cell that will represent this. Chat button. What I want is that I don't want to go into Excel wing slide to click the button over there. That is not good user interface.

I want to have a button here, I want to click this button. And when I click this button, I want my script to run and the script to run. I wanted to send in my question from here, from my prompt window. First I [00:09:00] want to send it into Gemini that it can gimme an answer and then I will put the question here that we can see what question was asked.

And then the bot answer should be here just by clicking this button and typing something here. And it should answer questions based on what we have in the data. This is the raw data. And I want to answer based on that one. Let us change this sheet to chat. And now let's open up Excel wing slide. It's time for us to, to code this out.

Starting with Excel wing slide, let us clean away this part. Let us clean away all these, we don't need that. This one we will need. Let's see. Excel wing slide. We will need [00:10:00] seaborne. We don't need pandas. We will need mpa. Don't need date, time. Don't need. And then we will need something else.

We'll need import os. Maybe I deleted that. Import Jason. We'll need data as well in order to process the data that comes from the API and then we need to have requests. Okay, no module named requests fine. We're going into requirements type in requests here.

Okay. And then I will change this one. I will. This is a decorator and the decorator. I will put in argument called button equals two, and then I will place in here chat button chat. [00:11:00] And then let's exclamation mark C five. Okay. I'll explain what this means, but this is basically the syntax. I will just remove this part as well.

We don't need that one. We don't need this. Okay. Yeah, this is all we need for now. First, in order to, for this I want to change this to something else, I would call it chat. Okay. It's a chatting function now. First, this one, we need to change its name. And to the left here you can see its name and we should give it the name of chat button.

And then if I take, okay. Chat underscore button enter. Okay, if I click something else, I click back here. It says chat button, underscore button. Okay. What it does is that okay, button equals to this chat Button. ~~it'll ~~it'll connect to this one.

**Kokchun Giang-4:** The chat button will connect to [00:12:00] this one, and the chat here means this sheet that is called chat. The sheet name. And then we have exclamation mark and C five means this cell should be marked. In order to mark this cell C five ~~and ~~and refer to the C five, what happens? It needs to connect to the C five.

We right click this one and we do hyperlink. And when we do hyperlink, we choose this document here and this is the chat. And you choose the cell reference. I'll choose C five. Okay. If I click okay here, let's just try, if I click this one, you can see that C five is marked. If I move this, let me see If I move this here and I click this one, you can see C five is connected.

Let me move back. ~~Move it back, ~~and let me just do a tester [00:13:00] print test. Or I can say button clicked to see that it works. I clicked this button. You can see Oh, button clicked. Oh, ~~so cool. ~~Cool. This is the first cool thing. What we should do here is that we have. I will clean this up actually as well.

You don't need this one sheets active. I will call this chat sheet, ~~remove that one, ~~and then I'll have copy this one shift out, down button the copy. I'll call this data sheet. Book sheets. Let's see, sheets off. Data. Now I have two sheets and then I will have prompt cell is this cell here is B five.

Prompt cell is [00:14:00] chat sheet of B five. I'll copy this one and I will have a. ~~Question, cell ~~question Cell is, let's see, B eight. B eight, and I will copy this again. I will have bot cell, pot cell is b. 10. Okay, let's try this we can have question. Cell dot value equals to prompt cell value. I want to move the question down here, right?

Tell me how many Pokemons are in [00:15:00] the data set, for example. If I click this one, you can see it jumps down here, but then afterwards when I do that, I want to clear out my question. To clear out the question, we can just do prompt sell. Dot value equals to nothing. If I type this here, you can see it's cleared and the question goes down here.

If I say hello here, I run this one. And you can see it jumps down here and it clears the cell. Okay. This is just some basic logic and then what is left is that we need to~~ read we need to, ask ~~put in the question into the LLM together with the data into the LLM, right? And we haven't even started coding the LLM out yet, that, that will be fun.

But let's start with reading in the data. That is a good thing to do. Def [00:16:00] read data and we'll put in the data sheet here. DF equals to data sheet. The data sheet is~~ is a ar ~~an argument that you send in. The data sheet dot range. I will start with a one. A one. Here is this one, and then I will do dot expand.

It'll expand. It'll expand~~ to ~~to I will show you how dot options PD data frame. Okay. I have pandas. Okay. And then dot value. This is a way to make this into panels data frame. Then we return the F. Let's just try it out. Not only I will clear the cell, but I will also read the data to see that it works.

Read [00:17:00] data, I'll call this df equals the, let's see, DF equals the read data of let's see, chat data sheet. I will say I need to send in and then I will just print DF and see that it works. I can do, yeah, print the F and then I will click this button here. Yes, you can see the data here. 7 21, 800 rows.

Okay, Good. we have read the data by just sending in the data sheet. Now what's left is our LLM. Def ask LLM. And this should take in a prompt and it should take in the data sheet because we need the, we need to take the data. Basically this this one here, we move it down here. We [00:18:00] need to, then I cannot print the F because it doesn't exist.

Basically it's that we want to get the prompt inside here. We want to get in the data sheet and then we get the df. Okay. What we do is that we need to construct our prompt. Prompt equals two, and. F stringing multi-line f stringing. There, there's an issue with coloring here, but it doesn't, it's okay.

I'll show you what the issue is soon. You are an expert in Pokemons, and you can see here there's an issue here. With coloring, but it's no problems. It's a multi-line string answer based on this data set. And we provide it with df. This is an F string, right? We can put in the DF variable here without problems.[00:19:00] 

Make the answer short. This is the question. And you send in the prompt. The prompt is what you get from here. You send it in here, and then you send in the df, which you get Here. ~~it can answer ba you get, you give it the context so that ~~it can answer based on this one. Okay? And then the idea is that, we need to for generative ai for Google, generative ai there, there's actually a library called Google Gen ai. And which you can use in Python. However, it doesn't work ~~in ~~in IDE and Excel Wings light because there is no something called ~~was ~~web assembly. There isn't this package available.

It's not possible to use ~~that ~~that interface. I usually use pedantic ai. I [00:20:00] like that one a lot. But ~~that is not, ~~even, that is not possible either. What we need to do is to work with the pure API, the API and do an API request. And in order to do that, let's move on to our browser.

I'll show you in Google AI Studio, you have this. Code here, right? This is a curl code. And curl is basically you can do this in your terminal if you're in Linux based systems such as or Unix based systems such as Mac. What you can do is that you copy this one, you go into for example, she GPT or Gemini, or should also work, take Gemini.

Translate this into Python code instead of curl. Okay? And you should get some code that you should be able to work with.

And [00:21:00] maybe there's something that needs to be cleaned up, but I have taken, I have already done this in my preparation, I will copy from my preparation, but I basically, this is the procedure that I did, I copied the curl code. Into Gemini or Shachi, I forgot. And ~~then ~~then I needed to change it somehow some, a little bit I think in order to make it work.

We go back to Excel here and I will copy that code in and you will get that code as well.

**Kokchun Giang-6:** This code means like this. We have after this three quotes here, ~~we have ~~this is a string, but you can see the color coding is wrong, but ~~the, it's, ~~it's not that the code is wrong. This is not strings down here, it's just the color coding. Here we start with API. Key is equal to O kti and v Gemini API key.

What you do is [00:22:00] that you go into here and you click environment variables. And in environment variables. You should add Gemini under API under key, and then you should paste in your values. When you paste it in, I just type in whatever and then I will click save and you can see, then you have your Gemini API key, and you have the value here.

And afterwards, let's go back here. And ~~what you can do is that we'll just ~~what you can do is to, ~~you can, when, ~~when you do O get nv, ~~you, ~~you type in Gemini API key and~~ it will, ~~you will get the API key and then you have this URL This is for~~ for get the, to get the, ~~to send the request, ~~the get ~~request.

And then, or the post requests, ~~sorry, not get requests. ~~You're sending in ~~the your post ~~what you want to say, the prompt, et cetera. And then what you will get back is you will get the response. This is the URL and then you have a head this you send in your API. [00:23:00] Here, this API key variable is sent in here.

And what you want to send in as content is JS N and then the payload. It looks like this. You have contents, you have role user, you have parts, and then you have text, and then you have the prompt. It's kind of nested. What you get back of data, it is nested data. What you need to do is that you need to.

Get it back ~~in, ~~in a strategic way. Here the response is the request of post the URL, the headers, and then the payload, which contains your prompt and you, you leave it for 60 seconds before timing out. And then this response dot raise for status is to check if there's any errors. And then you have data equals response to js ON.

But then you need to, this is the custom code that I've added. You need to extract the text from the top candidate. What you do is that you have here data dot get candidates and then you take out the [00:24:00] zero of it and then dot get content and then dot get parts and then zero of it and dot get text.

You need to go into, what I've done is that ~~I have, ~~I have printed this J out and then I have~~ taken out ~~gone into every trying to un nests it to get the actual text that I want. Then I return the text and basically like this. If I type in something, hello, and then I would run this one.

And then this one should run. Actually, we haven't asked the LLM yet, let's do that. We have prompt sell value before that one. We go into

bots. Let's see. Bot response equals to. Ask LM, and we put in prompt cell value. We [00:25:00] send in the prompt sell value, and we need to send in the data sheet. Data sheet down here. After we have done that, we get the bot response and we want to populate the bot. Cell dot value equals to bot response, as simple as that.

And then we clear the prompt cell. I save this one. I say hello, I run this one. And right now you can see there is an error because the error is that my API key was fake. The one that I just created. What you should do is that you should put in your real API key and I will post the video and I will do that.

**Kokchun Giang-7:** This is really cool. I have put in my Gemini API key, my real one. Now if I ask a question, how many Pokemons are [00:26:00] there ~~question? ~~I click on this button and here it says 800. Cool. Which are the electrical Pokemons that are legendary?

There are no electrical legendary Pokemon in the dataset. Okay. It's not very good model but~~ it ~~it looks into the dataset and let's ~~see if we can ~~see if there are, maybe there should be legendary. I'll ask if is Zap ~~does ~~legendary.

Yes, ~~septis ~~is legendary and ~~septis ~~is Electrical. ~~it's you, it's ~~it's an LLM. It's hallucinating. Sometimes it's not answering as good as you want it to be, but it's still answering. Based on this dataset, we can ask it like which are the strongest, which are the five strongest Pokemons? And list their stats.

Ask [00:27:00] this one. Yes. Here the concept of strongest Pokemons can be subjective as different stats contribute, blah, blah, blah. Okay. But here we have ~~new ~~two mega and. Blah, blah, blah. Okay, let's see. Going to data will make it like this total here is actually indication of the strength because it's the combination ~~of all the all, it's the sum ~~of all the other stats.

We can call this strength and save this one and going into chat, and then I will ask ~~you ~~this question again.

Okay, let me ask, actually, we can close down Excel Wings light over there. Paste this one and then I'll ask it again.

You can see it's different now based on the total stat. Actually ~~you need to you, ~~you, need to restart it in order to because I changed the data and also this, weirdly, it's remove my [00:28:00] thick lines. Okay. Which are the five strongest Pokemons here, as you can see, mute two. Mega mute two x.

And if we continue here, there's some more here. Quite cool. It's possible to chat with your data. With this quite simple application, I think, I guess it was simple. But~~ I hope ~~I hope that~~ you, ~~you can chat with whatever data you want, but make sure that since you're sending the data into an LLM that is on the web in this case Gemini don't use personal data.

And no company secrets, et Cetera. if you want to, chat with your personal data. ~~You should ~~you should have a local LLM which I can make a tutorial of that maybe in the future if there's a lot of interest. But I will leave that for the future.

**Kokchun Giang-8:** Wow, that was super cool. We managed to chat with our own data in Excel. ~~How about that? ~~Using Excel wing slide and some pi, we using Python to connect to the data frame, the data that we [00:29:00] had in Excel and to connect to Gemini, which we have the URL on the web and the API key, which we got them for free.

Just put it in and then work with it. ~~And then, ~~but ~~then ~~make sure that. Don't share your workbook ~~your Excel ~~however you want, since ~~you have your ~~you have your secrets inside, you don't want to share your API Key. ~~make sure that so if, ~~if you want to share it make sure to remove the API key first or use the adding scope.

There's different scopes. There's adding scope and there's the other one is the workbook scope. It's ~~tied ~~tied to the workbook, ~~but ~~the one that I shared to you, ~~it ~~doesn't have the API key. ~~So, ~~but you can take a look into ~~my ~~my Excel worksheet. ~~And ~~you can see how I built it, and you can see the code inside there as well.

The Excel wings, like the Python code. ~~And, but ~~it is fun to ~~co ~~combine excel with Python and to combine with LS that we're getting used to in ~~in the ~~daily life. ~~So. ~~[00:30:00] It's quite fun to just be able to ask your data different stuffs. ~~So you don't need to manually, ~~you don't need to manually process it with pandas or something you can share it with people that not very familiar with Python.

Quite fun. Yes, and I hope that you learned a lot from this video and see you in the next one. Bye.

