import { useState } from "react";
import Login from "./components/Login.jsx";
import Dashboard from "./components/Dashboard.jsx";
import ThemeToggle from "./components/ThemeToggle.jsx";

function hasToken() {
  return !!localStorage.getItem("upi_token");
}

export default function App() {
  const [authed, setAuthed] = useState(hasToken());

  return (
    <>
      {authed ? <Dashboard onLoggedOut={() => setAuthed(false)} /> : <Login onAuthed={() => setAuthed(true)} />}
      <ThemeToggle />
    </>
  );
}
