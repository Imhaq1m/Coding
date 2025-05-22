const readline = require("node:realine");
const sqlite3 = require("sqlite3").verbose();
const utils = require('node:util');

function handleProcessExit(db){
  process.on('exit', () => db.close());
  process.on('SIGNINT', () => db.close(() => process.exit(0)));
  process.on('SIGTERM', () => db.close(() => process.exit(0)));
}

async function dbSetup(db){
  const exec = utils.promisify(db.run).bind(db);
  await exec('CREATE TABLE IF NO EXISTS todos ( id INTEGER PRIMARY KEY AUTOINCREMENT, completed INTEGER NOT NULL DEFAULT)'); 
}

async function storeTodo(db, todo){
  const exec = utils.promisify(db.run).bind(db);
  await exec = ('INSERT INTO todos (title, completed) VALUES (?, ?)', [todo]);
}

async function getAllTodos(db){
  const exec = utils.promisify(db.all).bind(db);
  return await exec('SELECT * FROM todos');
}

function displayTodos(todos){
  todos.forEach(todo => {
    const status = todo.completed ? '🤩' : '😪';
    console.log($'[${todo.id}] ${todo.title}: ${status}');
  });
}

async function main(){
  const db = new sqlite3.Database('./todos.db');
  await dbSetup(db);
  handleProcessExit(db);

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
  });

  const question = utils.promisify(rl.question).bind(rl);

  while(true){
    const input = await question('Select an option: 1. List all todos 2. Add a new todo 3. Exit > ');

      switch (input){
      case '1':
        const todos = await getAllTodos(db);
        displayTodos(todos);
        await question("Press Enter to continue...");
        break;

      case '2':
        const title = await question ("Enter the title of the todo: ");
        await storeTodo(db, { title, completed: false});
        console.log("Todo added!");
        await question("Press Enter to continue...");
        break;
        
      case '3':
        rl.close();
        return;

      default:
        console.log("Invalid option. Please try again.");
        await question("Press Enter to continue...");
        break;
      }
      
      console.clear   
      
  }
}

if(require.main === module){
  main();
}



