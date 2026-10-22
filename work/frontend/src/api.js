const BASE_URL = "/api";

function getToken() {
  return localStorage.getItem("upi_token");
}

export function setToken(token) {
  localStorage.setItem("upi_token", token);
}

export function clearToken() {
  localStorage.removeItem("upi_token");
}

async function request(path, options = {}) {
  const headers = options.headers || {};
  const token = getToken();
  if (token) headers["Authorization"] = `Bearer ${token}`;

  let res;
  try {
    res = await fetch(`${BASE_URL}${path}`, { ...options, headers });
  } catch (_) {
    throw new Error("Can't reach the server. Check that the backend is running, then try again.");
  }
  if (res.status === 401 && token && !path.startsWith("/auth/login")) {
    clearToken();                      // session expired: back to the login screen
    window.location.reload();
    throw new Error("Your session ended. Please log in again.");
  }
  if (res.status >= 500) {
    throw new Error("Something went wrong on our side. Please try again in a moment.");
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch (_) {}
    throw new Error(detail);
  }
  return res.json();
}

export const api = {
  register: (email, password, display_name) =>
    request("/auth/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password, display_name }),
    }),

  login: (email, password) => {
    const form = new URLSearchParams();
    form.set("username", email);
    form.set("password", password);
    return request("/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: form.toString(),
    });
  },

  me: () => request("/auth/me"),

  uploadHistory: (file) => {
    const form = new FormData();
    form.append("file", file);
    return request("/upload/history", { method: "POST", body: form });
  },

  dashboard: () => request("/me/dashboard"),

  recentTransactions: (limit = 10) => request(`/me/transactions?limit=${limit}`),

  retrainNow: () => request("/me/retrain", { method: "POST" }),

  driftHistory: () => request("/me/drift-history"),
};
