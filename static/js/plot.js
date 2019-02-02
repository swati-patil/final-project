
  
  function init() {
     filterCharts()
    // d3.json("/all").then((data) => {
    //   filter_data(data);
    // });
  
  }

  function filterCharts() {
      var category = d3.select("#cat-select").node().value;
      if (category == "Travel"){
          var cat_filter = 1;
      } else if (category == "Sports & Outdoors"){
          var cat_filter = 2;
      } else if (category == "Health"){
          var cat_filter = 3;
      } else if (category == "Computers & Technology"){
          var cat_filter = 4;
      } else if (category == "Politics & Social Sciences"){
          var cat_filter = 5;
      } else if (category =="Education & Teaching"){
          var cat_filter = 6;
      } else if (category =="New"){
          var cat_filter = 7;
      } else if (category == "Childrens Books"){
          var cat_filter = 8;
      } else if (category == "Biographies & Memoirs"){
          var cat_filter = 9;
      } else if (category == "Literature & Fiction"){
          var cat_filter = 10;
      } else if(category == "Crafts"){
          var cat_filter = 11;
      } else if (category == "Cookbooks"){
          var cat_filter = 12;
      } else if (category == "Science Fiction & Fantasy"){
          var cat_filter = 13;
      } else if (category == "Reference"){
          var cat_filter = 14;
      } else if (category == "Arts & Photography"){
          var cat_filter = 15;
      } else if (category == "Humor & Entertainment"){
          var cat_filter = 16;
      } else if (category == "Medical Books"){
          var cat_filter = 17;
      } else if (category == "History"){
          var cat_filter = 18;
      } else if (category == "Science & Math"){
          var cat_filter = 19;
      } else if (category == "Business & Money"){
          var cat_filter = 20;
      } else if (category == "Teen & Young Adult"){
          var cat_filter = 21;
      } else if (category == "Comics & Graphic Novels"){
          var cat_filter = 22;
      } else if (category == "Engineering & Transportation"){
          var cat_filter = 23;
      } else if (category == "Calendars"){
          var cat_filter = 24;
      } else if (category == "Self-Help"){
          var cat_filter = 25;
      } else if (category == "Christian Books & Bibles"){
          var cat_filter = 26;
      } else if (category == "Parenting & Relationships"){
          var cat_filter = 27;
      } else if (category == "Religion & Spirituality"){
          var cat_filter = 28;
      } else if (category == "Law"){
          var cat_filter = 29;
      } else if (category == "Mystery"){
          var cat_filter = 30;
      } else if (category == "Gay & Lesbian"){
          var cat_filter = 31;
      } else if (category == "Romance"){
          var cat_filter = 32;
      } else if (category == "Crafts & Sewing"){
          var cat_filter = 33;
      } else if (category == "Exterior Accessories"){
          var cat_filter = 34;
      }

      var price = d3.select("#price_input").node().value;
      //var price = document.getElementById("#price").value;
    
    console.log(cat_filter)
    //var url = "/predict/"+price+"/"+cat_filter;
  
    //console.log(url);
    //d3.json(url).then((data) => {
        //console.log(data);
        //console.log(data.category);
        //console.log(data.predicted_val);
        //prediction = data.predicted_val
        document.getElementById("Category_result").innerHTML = category;
        document.getElementById("Price_result").innerHTML = price;
        //document.getElementById("Predicted_result").innerHTML = prediction;
    //    });
    }


var modal = document.getElementById('myModal');


var img = document.getElementById('myImg');
var modalImg = document.getElementById("img01");
var captionText = document.getElementById("caption");
img.onclick = function(){
  modal.style.display = "block";
  modalImg.src = this.src;
  captionText.innerHTML = this.alt;
}


var span = document.getElementsByClassName("close")[0];


span.onclick = function() { 
  modal.style.display = "none";
}
  
   init();