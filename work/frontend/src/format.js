const inr = new Intl.NumberFormat("en-IN", { maximumFractionDigits: 0 });
export const rupees = (n) => `₹${inr.format(Math.round(n ?? 0))}`;
export const day = (s) =>
  s ? new Date(s.replace(" ", "T")).toLocaleDateString("en-IN", { day: "numeric", month: "short", year: "numeric" }) : "—";
