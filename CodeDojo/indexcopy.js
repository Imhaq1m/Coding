const readline = require("node:readline");
const sqlite3 = require("sqlite3").verbose();
const utils = require("node:util");

function handleProcessExit(db) {
  process.on("exit", () => db.close());
  process.on("SIGINT", () => {
    db.close(() => {
      console.log("\nDatabase closed. Exiting...");
      process.exit(0);
    });
  });
  process.on("SIGTERM", () => {
    db.close(() => {
      console.log("\nDatabase closed. Exiting...");
      process.exit(0);
    });
  });
}

async function dbSetup(db) {
  const exec = utils.promisify(db.run).bind(db);
  await exec(`
    CREATE TABLE IF NOT EXISTS todos (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      title TEXT NOT NULL,
      completed INTEGER NOT NULL DEFAULT 0
    )
  `);
}

async function storeTodo(db, todo) {
  const exec = utils.promisify(db.run).bind(db);
  await exec("INSERT INTO todos (title, completed) VALUES (?, ?)", [todo.title, todo.completed ? 1 : 0]);
}

async function getAllTodos(db) {
  const exec = utils.promisify(db.all).bind(db);
  return await exec("SELECT * FROM todos");
}

function displayTodos(todos) {
  if (todos.length === 0) {
    console.log("No todos found.");
    return;
  }

  todos.forEach((todo) => {
    const status = todo.completed ? "🤩" : "😪";
    console.log(`${todo.id} ${todo.title}: ${status}`);
  });
}

async function main() {
  const db = new sqlite3.Database("./todos.db");
  try {
    await dbSetup(db);
    handleProcessExit(db);

    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
    });

    const question = utils.promisify(rl.question).bind(rl);

    while (true) {
      console.clear();
      const input = await question(
        "Select an option:\n1. List all todos\n2. Add a new todo\n3. Exit\n> "
      );

      switch (input) {
        case "1":
          const todos = await getAllTodos(db);
          displayTodos(todos);
          await question("Press Enter to continue...");
          break;

        case "2":
          const title = await question("Enter the title of the todo: ");
          await storeTodo(db, { title, completed: false });
          console.log("✅ Todo added!");
          await question("Press Enter to continue...");
          break;

        case "3":
          rl.close();
          db.close(() => {
            console.log("👋 Goodbye!");
            process.exit(0);
          });
          return;

        default:
          console.log("❌ Invalid option. Please try again.");
          await question("Press Enter to continue...");
          break;
      }
    }
  } catch (err) {
    console.error("Error:", err.message);
    db.close(() => process.exit(1));
  }
}

if (require.main === module) {
  main();
}
