import 'package:flutter/material.dart';
import 'package:html2md/html2md.dart' as html2md;
import 'package:flutter_markdown/flutter_markdown.dart';

void main() => runApp(MyApp());

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Cortex Main',
      debugShowCheckedModeBanner: false,
      theme: ThemeData( primarySwatch: Colors.blue),
        routes: <String, WidgetBuilder>{
          '/': (BuildContext context) {return HomePage(title: 'Main Page');},
          '/about': (BuildContext context) { return Scaffold(appBar: AppBar(title: const Text('wel'),),); },
        }
    );
  }
}

/*
Scaffold (
  appBar: AppBar(
          title: const Text('Home Route'),
          )
)
 */

final GlobalKey<ScaffoldState> scaffoldKey = GlobalKey<ScaffoldState>();
final SnackBar snackBar = const SnackBar(content: Text('Showing Snackbar'));

void openPage(BuildContext context) {
  Navigator.push(context, MaterialPageRoute(
    builder: (BuildContext context) {
      return Scaffold(
        appBar: AppBar(
          title: const Text('Next page'),
        ),
        body: const Center(
          child: Text(
            'This is the next page',
            style: TextStyle(fontSize: 24),
          ),
        ),
      );
    },
  ));
}

class HomePage extends StatelessWidget {
  HomePage({Key key, this.title}) : super(key: key);
  final String title;
  String text_sample = "Initial Text";
  static String html = '<h1>This is heading 1</h1> <h2>This is heading 2</h2><h3>This is heading 3</h3><h4>This is heading 4</h4><h5>This is heading 5</h5><h6>This is heading 6</h6><p><img alt="Test Image" src="https://i.ytimg.com/vi/RHLknisJ-Sg/maxresdefault.jpg" /></p>';
  String markdown = html2md.convert(html);

  final Widget button_1 =  IconButton(icon: const Icon(Icons.add_alert), onPressed: () { scaffoldKey.currentState.showSnackBar(snackBar);});

  @override
  Widget build(BuildContext context) {
    return
      Scaffold(
        key: scaffoldKey,
        drawer: Drawer( child: ListView(
          padding: EdgeInsets.zero,
          children: <Widget>[
            UserAccountsDrawerHeader(
              accountName: Text("Naveen Mysore"),
              accountEmail: Text("navimn1991@gmail.com"),
              currentAccountPicture: CircleAvatar(
                backgroundColor: Colors.white,
                child: Text(
                  "N",
                  style: TextStyle(fontSize: 40.0),
                ),
              ),
            ),
            ListTile(
              title: Text('Data'),
              onTap: () {
                // Update the state of the app
                Navigator.pop(context);
                Navigator.of(context).push(MaterialPageRoute(
                    builder: (BuildContext context) => Scaffold (
                      appBar: AppBar (
                        title: const Text('Next page'),
                      ),
                      body: const Center(
                        child: Text(
                          'This is the next page',
                          style: TextStyle(fontSize: 24),
                        ),
                      ),
                    )
                ));
              },
            ),
            ListTile(
              title: Text('Geo Spatial'),
              onTap: () {
                // Update the state of the app.
                Navigator.pop(context);
              },
            ),
          ],
        )),

        appBar: AppBar(
            title: const Text('Cortex'),
            actions: <Widget>[
              button_1,
              IconButton(icon: const Icon(Icons.navigate_next), onPressed: () { openPage(context);})
              ]
          ),
        body: new Center(
          child: new Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: <Widget>[
              new MarkdownBody(
                data: markdown,
              )
            ],
          ),
        ),
      );
  }
}