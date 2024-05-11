$(document).ready(function () {
    // Function to fetch info from Flask backend
    function fetchInfo(selectedCurrency) {
        if (!selectedCurrency) {
            // No currency selected, do nothing
            return;
        }
        var nonUSDCurrency = selectedCurrency.replace('USD', '');

        $.ajax({
            url: "/currency_info",
            type: "GET",
            data: {
                currency: nonUSDCurrency
            },
            success: function (response) {
                // Show submit button if a valid currency pair is selected
                $("#submitButton").prop("disabled", !selectedCurrency);

                // If a valid currency pair is selected, update info display
                $("#infoDisplay").show();
                // Update HTML with fetched data
                $("#currency").text(response.currency);
                $("#start").text(response.start);
                $("#end").text(response.end);
                $("#sum").text(response.sum);
                $("#avg").text(response.avg);
                $("#risk").text(response.risk);
                // Show the "Get 10 News" button
                $("#getNewsButton").show();
                $("#getNewsButton").text("Latest " + response.currency + " or USD News");

                $("#askAdvisor").show();
            },
            error: function (xhr, status, error) {
                console.error("Error fetching data:", error);
            }
        });
    }

    // Listen for change event on select element
    $("#currencyPair").change(function () {
        var selectedCurrency = $(this).val();
        if (selectedCurrency) {
            fetchInfo(selectedCurrency); // Fetch and display info
        } else {
            $("#submitButton").prop("disabled", !selectedCurrency);
            var infoDisplay = document.getElementById("infoDisplay");
            infoDisplay.style.display = selectedCurrency ? "block" : "none";
        }
    });

    // Initial fetch when page loads
    fetchInfo($("#currencyPair").val());


    // Event listener for click event on the 'ask advisor' button
    $("#askAdvisor").click(function () {
        var chatModal = new bootstrap.Modal(document.getElementById('chatModal'), { backdrop: 'static' });
        chatModal.show();

        $("#userInput").keypress(function (e) {
            if (e.which === 13) {
                $("#sendMessageButton").click();
            }
        });

        // Populate the chat modal with the news article details
        $("#chatLog").html("<div class='message bot'>" + "Hi, I am your personal AI investment Advisor! Feel free to ask me anything! </div>");

        // hide analysis-buttons
        $(".analysis-buttons").hide();

        $("#sendMessageButton").click(function () {
            var userMessage = $("#userInput").val();
            // Send userMessage to GPT API and display response
            if (!userMessage) {
                return
            }
            // Append the user message to the chat log
            $("#chatLog").append("<div class='message user'>" + userMessage + "</div>");

            // Clear the user input field
            $("#userInput").val("");

            $("#sendMessageButton").prop("disabled", true);


            $.ajax({
                url: "/advise",
                type: "GET",
                data: {
                    prompt: userMessage,
                },
                success: function (response) {
                    // Append the bot response to the chat log
                    $("#chatLog").append("<div class='message bot'>" + response + "</div>");
                },
                error: function (xhr, status, error) {
                    console.error("Error sending message:", error);
                    $("#chatLog").append("<div class='message bot'>Error sending message:" + error + "</div>");
                },
                complete: function () {
                    $("#sendMessageButton").prop("disabled", false);

                    // Scroll to the bottom of the chat log
                    $("#chatLog").scrollTop($("#chatLog")[0].scrollHeight);
                }
            });


        });



    });
    $("#getNewsButton").click(function () {
        $('#getNewsButton').prop("disabled", true);
        $("#loadingnews").show();
        var selectedCurrency = $("#currencyPair").val();
        var nonUSDCurrency = selectedCurrency.replace('USD', '');
        //disable the button
        if (nonUSDCurrency) {
            $.ajax({
                url: "/news",
                type: "GET",
                data: {
                    currency: nonUSDCurrency
                },
                success: function (response) {
                    // Clear existing news articles if any
                    $("#getNewsButton").prop("disabled", false);
                    $("#newsTableBody").empty();
                    $("#loadingnews").hide();
                    $("#newsTable").show();
                    // Iterate over each news news in the response
                    $.each(response, function (index, news) {


                        // Create a new row for the news news
                        var row = $("<tr>");
                        var indexColumn = $("<td>").text(index + 1);
                        // Add columns for each key in the JSON
                        var dateColumn = $("<td>").text(news.date);
                        var headlineColumn = $("<td>").text(news.headline);
                        var briefColumn = $("<td>").text(news.brief);
                        var scoreColumn = $("<td>").text(news.score);
                        var currencyColumn = $("<td>").text(news.currency);
                        // Create the link
                        var linkColumn = $("<td>");
                        var link = $("<a>").text("Ask");
                        link.attr("href", "#"); // Set the href attribute to whatever link you want
                        linkColumn.append(link);

                        // Append columns to the row
                        row.append(indexColumn, dateColumn, currencyColumn, headlineColumn, briefColumn, scoreColumn, linkColumn);

                        // Append the row to the table body
                        $("#newsTableBody").append(row);
                    });
                    // Add event listener to "Ask" links
                    $("#newsTableBody a").click(function (e) {
                        e.preventDefault(); // Prevent default link behavior
                        var chatModal = new bootstrap.Modal(document.getElementById('chatModal'), { backdrop: 'static' });
                        chatModal.show();

                        $("#userInput").keypress(function (e) {
                            if (e.which === 13) {
                                $("#sendMessageButton").click();
                            }
                        });

                        // Fetch relevant information based on the clicked news article and display it in the chat modal
                        var date = $(this).closest("tr").find("td:eq(1)").text();
                        var currency = $(this).closest("tr").find("td:eq(2)").text();
                        var headline = $(this).closest("tr").find("td:eq(3)").text();
                        var brief = $(this).closest("tr").find("td:eq(4)").text();
                        var score = $(this).closest("tr").find("td:eq(5)").text();

                        // Populate the chat modal with the news article details
                        $("#chatLog").html("<div class='message bot'>" + "<p><strong>Date:</strong> " + date + "</p>" +
                            "<p><strong>Headline:</strong> " + headline + "</p>" +
                            "<p><strong>Brief:</strong> " + brief + "</p>" +
                            "<p><strong>Score:</strong> " + score + "</p>" + "</div>");

                        $(".analysis-buttons").show();


                        $("#sendMessageButton").click(function () {
                            var userMessage = $("#userInput").val();
                            // Send userMessage to GPT API and display response
                            if (!userMessage) {
                                return
                            }
                            // Append the user message to the chat log
                            $("#chatLog").append("<div class='message user'>" + userMessage + "</div>");

                            // Clear the user input field
                            $("#userInput").val("");

                            $("#sendMessageButton").prop("disabled", true);

                            // hide analysis-buttons
                            $(".analysis-buttons").hide();

                            $.ajax({
                                url: "/askAI",
                                type: "GET",
                                data: {
                                    prompt: userMessage,
                                    h: headline,
                                    b: brief,
                                    c: currency,
                                },
                                success: function (response) {
                                    // Append the bot response to the chat log
                                    $("#chatLog").append("<div class='message bot'>" + response + "</div>");
                                },
                                error: function (xhr, status, error) {
                                    console.error("Error sending message:", error);
                                    $("#chatLog").append("<div class='message bot'>Error sending message:" + error + "</div>");
                                },
                                complete: function () {
                                    $("#sendMessageButton").prop("disabled", false);
                                    // show analysis-buttons
                                    $(".analysis-buttons").show();

                                    // Scroll to the bottom of the chat log
                                    $("#chatLog").scrollTop($("#chatLog")[0].scrollHeight);
                                }
                            });
                            $("#getNewsButton").prop("disabled", false);
                        });

                        $("#anaNews").click(function () {
                            var userMessage = 'Analyze the news';
                            // Send userMessage to GPT API and display response

                            // Append the user message to the chat log
                            $("#chatLog").append("<div class='message user'>" + userMessage + "</div>");

                            // Clear the user input field
                            $("#userInput").val("");

                            $("#sendMessageButton").prop("disabled", true);

                            // hide analysis-buttons
                            $(".analysis-buttons").hide();

                            $.ajax({
                                url: "/askAI",
                                type: "GET",
                                data: {
                                    prompt: userMessage,
                                    h: headline,
                                    b: brief,
                                    c: currency,
                                },
                                success: function (response) {
                                    // Append the bot response to the chat log
                                    $("#chatLog").append("<div class='message bot'>" + response + "</div>");
                                },
                                error: function (xhr, status, error) {
                                    console.error("Error sending message:", error);
                                    $("#chatLog").append("<div class='message bot'>Error sending message:" + error + "</div>");
                                },
                                complete: function () {
                                    $("#sendMessageButton").prop("disabled", false);
                                    // show analysis-buttons
                                    $(".analysis-buttons").show();

                                    // Scroll to the bottom of the chat log
                                    $("#chatLog").scrollTop($("#chatLog")[0].scrollHeight);
                                }
                            });
                        });

                        $("#anaCurr").click(function () {
                            var userMessage = 'Analyze the currency';
                            // Send userMessage to GPT API and display response

                            // Append the user message to the chat log
                            $("#chatLog").append("<div class='message user'>" + userMessage + "</div>");

                            // Clear the user input field
                            $("#userInput").val("");

                            $("#sendMessageButton").prop("disabled", true);

                            // hide analysis-buttons
                            $(".analysis-buttons").hide();

                            $.ajax({
                                url: "/askAI",
                                type: "GET",
                                data: {
                                    prompt: userMessage,
                                    h: headline,
                                    b: brief,
                                    c: currency,
                                },
                                success: function (response) {
                                    // Append the bot response to the chat log
                                    $("#chatLog").append("<div class='message bot'>" + response + "</div>");
                                },
                                error: function (xhr, status, error) {
                                    console.error("Error sending message:", error);
                                    $("#chatLog").append("<div class='message bot'>Error sending message:" + error + "</div>");
                                },
                                complete: function () {
                                    $("#sendMessageButton").prop("disabled", false);
                                    // show analysis-buttons
                                    $(".analysis-buttons").show();

                                    // Scroll to the bottom of the chat log
                                    $("#chatLog").scrollTop($("#chatLog")[0].scrollHeight);
                                }
                            });
                        });

                    });
                },
                error: function (xhr, status, error) {
                    console.error("Error fetching news:", error);
                    $("#getNewsButton").prop("disabled", false);
                }
            });
        }
    });
});
